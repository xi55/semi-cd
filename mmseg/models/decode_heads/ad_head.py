# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from typing import List, Optional, Tuple
from torch import Tensor
from mmseg.utils import SampleList
from ..losses import accuracy
from mmcv.cnn import (Linear, build_activation_layer)



class LinearClsHead(BaseModule):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
    # def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
    #     """The process before the final classification head.

    #     The input ``feats`` is a tuple of tensor, and each tensor is the
    #     feature of a backbone stage. In ``LinearClsHead``, we just obtain the
    #     feature of the last stage.
    #     """
    #     # The LinearClsHead doesn't have other module, just return after
    #     # unpacking.
    #     return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        # pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(feats.permute(0, 2, 3, 1))
        
        cls_score = self.sigmoid(cls_score)
        
        return cls_score

@MODELS.register_module()
class ADHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.post = PostBlock(in_channels=self.in_channels[-1],
                              in_index= self.in_index[-1],
                              channels=512,
                              num_classes = 5)
        self.clsConv = ConvModule(
            self.in_channels[-1],
            self.in_channels[-1],
            8,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.post = LinearClsHead(num_classes=2, in_channels=768, init_cfg=None)
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)

        post_out = inputs[-1]
        post_out = self.clsConv(post_out)
        post_out = self.post(post_out)

        # print(post_out.shape)
        # print(post_out)
        return [output, post_out.permute(0, 3, 1, 2)]
        # print(output.shape)
        # return [output, 0]

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        
        seg_post = seg_logits[1]

        seg_logits  = seg_logits[0]

        

        label_post = torch.zeros((seg_post.shape[0], seg_post.shape[2], seg_post.shape[3]), dtype=int)
        for i in range(0, seg_post.shape[0]):
            flag = torch.any(seg_label[i] == 2).item() and torch.any(seg_label[i] == 3).item() and torch.any(seg_label[i] == 4).item()
            if flag:
                label_post[i,:,:] = 1

        label_post = label_post.cuda()


        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label.long(),
                    weight=seg_weight)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight)
                

        
        criterion = nn.CrossEntropyLoss()
        loss['loss_post'] = criterion(
                    seg_post,
                    label_post.long())
        
        loss['acc_post'] = accuracy(
            seg_post, label_post, ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # print(loss)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_post = seg_logits[1]
        
        seg_logits  = seg_logits[0]
        
        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return [seg_logits, seg_post]

class Mlp(BaseModule):
    def __init__(self, 
                 embed_dims=4096,
                 feedforward_channels=1024,
                 num_fcs=2):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        layers = []
        in_channels = embed_dims
        
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    build_activation_layer(dict(type='GELU')), nn.Dropout(0.3)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, 2))
        layers.append(nn.Dropout(0.3))
        self.layers = Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class PostBlock(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvModule(
                self.num_classes,
                self.num_classes,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
        self.flatten = nn.Flatten()
        
        self.mlp1 = Mlp(embed_dims = self.num_classes, feedforward_channels = 20)
        self.mlp2 = Mlp(embed_dims=8192, feedforward_channels=16384)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 3, 1)

        # print(x.shape)
        feats = self.mlp1(x)
        # print(feats.shape)
        # feats = []
        # for i in range(0, x.shape[1]):
        #     feat = x[:, i, :].unsqueeze(1)
        #     feats.append(self.mlp1(feat))

        # feats = torch.concat(feats,dim=1)
        feats = self.flatten(feats)
        # print(feats.shape)
        res = self.mlp2(feats)
        res = self.softmax(res)
        # print(torch.max(res))
        return res   