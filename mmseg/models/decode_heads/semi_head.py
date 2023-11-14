# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from torch import Tensor
from typing import List, Tuple
import matplotlib.pyplot as plt

@MODELS.register_module()
class SemiHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):

        x1 = self._transform_inputs(inputs)
        # x2 = self._transform_inputs(inputs[1])

        output1 = self.scale_heads[0](x1[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output1 = output1 + resize(
                self.scale_heads[i](x1[i]),
                size=output1.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output1 = self.cls_seg(output1)

        # output2 = self.scale_heads[0](x2[0])
        # for i in range(1, len(self.feature_strides)):
        #     # non inplace
        #     output2 = output2 + resize(
        #         self.scale_heads[i](x1[i]),
        #         size=output2.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)

        # output2 = self.cls_seg(output2)
        # return [output1, output2]
        return output1
    

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.label_seg_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def display1(self, affinity, s):
        affinity = affinity.permute(1, 2, 0)
        affinity = affinity.cpu().numpy()
        plt.imshow(affinity)
        plt.title(s)
        plt.colorbar()
        plt.show()

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
        # print(len(seg_logits))
        # seg_logits = seg_logits
        # print(batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)

        # self.display1(seg_label[0], batch_data_samples[0].seg_map_path.split('\\')[-1])
        # print(torch.unique(seg_label))
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
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss
