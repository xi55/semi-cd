# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional
from mmseg.utils import (ConfigType, SampleList, OptConfigType, add_prefix)
import torch
from torch import Tensor
import torch.nn as nn
from mmseg.registry import MODELS
from .sll_semi_encoder_decoder import SllSemiEncoderDecoder
import matplotlib.pyplot as plt
from mmengine.structures import PixelData
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.structures import SegDataSample
from mmseg.models.utils import resize

@MODELS.register_module()
class SegSemiEncoderDecoder(SllSemiEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self, backbone, decode_head, neck: OptConfigType = None, pretrained: Optional[str] = None, *args, **kwargs):
        super().__init__(backbone = backbone, decode_head=decode_head, *args, **kwargs)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone_student = MODELS.build(backbone)
        self.backbone_teacher = MODELS.build(backbone)
        self._init_decode_head(decode_head)
        self._init_neck(neck)
        self.freeze(self.backbone_teacher)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        if not model.training:
            return
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_student = MODELS.build(decode_head)
        self.decode_teacher = MODELS.build(decode_head)
        self.align_corners = self.decode_student.align_corners
        self.num_classes = self.decode_student.num_classes
        self.out_channels = self.decode_student.out_channels
        self.freeze(self.decode_teacher)

    def _init_neck(self, neck: ConfigType):
        if neck is not None:
            self.neck_student = MODELS.build(neck)
            self.neck_teacher = MODELS.build(neck)
            self.freeze(self.neck_teacher)


    def extract_feat(self, inputs) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_s=None
        img_l, img_u, img_s = torch.split(inputs, self.backbone_inchannels, dim=1)

        feat_w = self.backbone_teacher(img_u)

        feat_fp = [nn.Dropout2d(0.5)(f) for f in feat_w]

        feat_s = self.backbone_teacher(img_s)

        feat_l = self.backbone_student(img_l)

        # if self.with_neck:
        #     feat_w_from[3] = self.neck_teacher(feat_w_from[3])
        #     feat_w_to[3] = self.neck_teacher(feat_w_to[3])


        # x.append(x_seg)
        return feat_w, feat_fp, feat_s, feat_l
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feat_w, feat_fp, feat_s, feat_l = self.extract_feat(inputs)
        w_logits_seg = self.decode_teacher.predict(feat_w, batch_img_metas,
                                              self.test_cfg)

        fp_logits_seg = self.decode_teacher.predict(feat_fp, batch_img_metas,
                                                    self.test_cfg)
        
        s_logits_seg = self.decode_teacher.predict(feat_s, batch_img_metas,
                                                    self.test_cfg)

        l_logits_seg = self.decode_student.predict(feat_l, batch_img_metas,
                                                    self.test_cfg)
        # print(batch_img_metas[0].keys())
        # self.display1(seg_logits_cd[0], batch_img_metas[0]['seg_map_path'].split('\\')[-1])
        
        return [w_logits_seg, fp_logits_seg, s_logits_seg, l_logits_seg]

    def display1(self, affinity, s):
        affinity = affinity.permute(1, 2, 0)
        affinity = affinity.cpu().numpy()
        plt.imshow(affinity)
        plt.title(s)
        plt.colorbar()
        plt.show()

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
            """Calculate losses from a batch of inputs and data samples.

            Args:
                inputs (Tensor): Input images.
                data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                    It usually includes information such as `metainfo` and
                    `gt_sem_seg`.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            feat_w, feat_fp, feat_s, feat_l = self.extract_feat(inputs)
            pseudo_label, mask = self.get_pseudo(feat_w, data_samples)
            losses = dict()

            loss_decode = self._decode_head_forward_train(feat_l,
                                                          feat_w,
                                                          feat_fp,
                                                          feat_s,
                                                          pseudo_label, mask,
                                                          data_samples)
            losses.update(loss_decode)

            
            # if losses['decode.loss_stu'] > 0.1:
            #     loss_cd_decode = self._cd_decode_head_forward_train([feat_from, feat_to], pseudo_label_cd, mask_cd, data_samples)
            #     losses.update(loss_cd_decode)
            # loss_cd_decode = self._cd_decode_head_forward_train([feat_from, feat_to], pseudo_label_cd, mask_cd, data_samples)
            # losses.update(loss_cd_decode)
            # if self.with_auxiliary_head:
            #     loss_aux = self._auxiliary_head_forward_train([feat_from, feat_to, feat_seg], data_samples)
            #     losses.update(loss_aux)

            return losses
    
    def _decode_head_forward_train(self, 
                                   feat_l,
                                   feat_w,
                                   feat_fp,
                                   feat_s,
                                   pseudo_label, mask,
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        
        loss_decode = self.decode_student.loss(feat_l, data_samples,
                                            self.train_cfg)
        # loss_stu = self.decode_teacher.loss_stu(feat_fp,
        #                                         feat_s,
        #                                         pseudo_label, mask)
        # loss_decode.update(loss_stu)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def ssl_postprocess_result(self,
                           seg_logits: list[Tensor],
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        w_logits_seg, fp_logits_seg, s_logits_seg, l_logits_seg = seg_logits
        batch_size, C, H, W = l_logits_seg.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                
                # i_seg_logits shape is 1, C, H, W after remove padding
                l_seg_logits = l_logits_seg[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                w_seg_logits = w_logits_seg[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        l_seg_logits = l_seg_logits.flip(dims=(3, ))
                        w_seg_logits = w_seg_logits.flip(dims=(3, ))
                    else:
                        l_seg_logits = l_seg_logits.flip(dims=(2, ))
                        w_seg_logits = w_seg_logits.flip(dims=(2, ))

                # resize as original shape
                res=[]
                for i_seg in [l_seg_logits, w_seg_logits]:
                    i_seg = resize(
                        i_seg,
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False).squeeze(0)
                    res.append(i_seg)
                l_seg_logits, w_seg_logits = res
            else:
                l_seg_logits = l_seg_logits[i]
                w_seg_logits = w_seg_logits[i]

            if C > 1:
                l_seg_pred = l_seg_logits.argmax(dim=0, keepdim=True)
                w_seg_pred = w_seg_logits.argmax(dim=0, keepdim=True)
            else:
                l_seg_logits = l_seg_logits.sigmoid()
                w_seg_logits = w_seg_logits.sigmoid()
                

                l_seg_pred = (l_seg_logits >
                              0.5).to(l_seg_logits)
                w_seg_pred = (w_seg_logits >
                              0.5).to(w_seg_logits)


            data_samples[i].set_data({
                'l_seg_pred':
                PixelData(**{'data': l_seg_pred}),
                'w_seg_pred':
                PixelData(**{'data': w_seg_pred})
            })

        return data_samples


    @torch.no_grad()
    def get_pseudo(self, inputs, data_samples: SampleList):
        w_logits_seg = self.decode_teacher.predict(inputs, data_samples, self.test_cfg)

        pseudo_label_seg = torch.softmax(w_logits_seg.detach(), dim=1)
        pseudo_mask, pseudo_label = torch.max(pseudo_label_seg, dim=1)
        mask = pseudo_mask.ge(0.95).float()
        return pseudo_label, mask
    


