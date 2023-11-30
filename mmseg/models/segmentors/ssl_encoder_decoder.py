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

@MODELS.register_module()
class SLLEncoderDecoder(SllSemiEncoderDecoder):
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
        if isinstance(inputs, Tensor):
            img_from, img_to, img_seg, img_from_strong, img_to_strong = torch.split(inputs, self.backbone_inchannels, dim=1)
        elif isinstance(inputs, list):
            img_from, img_to, img_seg = inputs

        feat_from = self.backbone_teacher(img_from)
        feat_to = self.backbone_teacher(img_to)

        feat_fp_from = [nn.Dropout2d(0.5)(f) for f in feat_from]
        feat_fp_to = [nn.Dropout2d(0.5)(t) for t in feat_to]

        stu_from = self.backbone_teacher(img_from_strong)
        stu_to = self.backbone_teacher(img_to_strong)

        # stu_from = self.backbone_student(img_from_strong)
        # stu_to = self.backbone_student(img_to_strong)

        feat_seg = self.backbone_student(img_seg)

        if self.with_neck:
            if isinstance(feat_from, tuple):
                feat_from = list(feat_from)
                feat_to = list(feat_to)
                stu_from = list(stu_from)
                stu_to = list(stu_to)
                feat_seg = list(feat_seg)

            feat_from[3] = self.neck_teacher(feat_from[3])
            feat_to[3] = self.neck_teacher(feat_to[3])
            
            feat_fp_from[3] = self.neck_teacher(feat_fp_from[3])
            feat_fp_to[3] = self.neck_teacher(feat_fp_to[3])

            stu_from[3] = self.neck_teacher(stu_from[3])
            stu_to[3] = self.neck_teacher(stu_to[3])

            feat_seg[3] = self.neck_student(feat_seg[3])

        # x.append(x_seg)
        return feat_from, feat_to, feat_seg, stu_from, stu_to, feat_fp_from, feat_fp_to
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feat_from, feat_to, feat_seg, stu_from, stu_to, feat_fp_from, feat_fp_to = self.extract_feat(inputs)
        seg_logits_from = self.decode_teacher.predict(feat_from, batch_img_metas,
                                              self.test_cfg)
        
        seg_logits_to = self.decode_teacher.predict(feat_to, batch_img_metas,
                                              self.test_cfg)

        seg_logits_stu_from = self.decode_teacher.predict(stu_from, batch_img_metas,
                                                    self.test_cfg)
        seg_logits_stu_to = self.decode_teacher.predict(stu_to, batch_img_metas,
                                                    self.test_cfg)

        seg_logits_seg = self.decode_student.predict(feat_seg, batch_img_metas,
                                                    self.test_cfg)
        
        seg_logits_cd = self.cd_decode_head.predict([feat_from, feat_to], batch_img_metas,
                                                    self.test_cfg)
        # print(batch_img_metas[0].keys())
        # self.display1(seg_logits_cd[0], batch_img_metas[0]['seg_map_path'].split('\\')[-1])
        
        return [seg_logits_from, seg_logits_to, seg_logits_seg, seg_logits_cd, seg_logits_stu_from, seg_logits_stu_to]

    def display1(self, affinity, s):
        affinity = affinity.permute(1, 2, 0)
        affinity = affinity.cpu().numpy()
        plt.imshow(affinity)
        plt.title(s)
        plt.colorbar()
        plt.show()

    def _stack_batch_from(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.i_seg_from_pred.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def _stack_batch_to(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.i_seg_to_pred.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

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
            # _, _, img_seg = torch.split(inputs, self.backbone_inchannels, dim=1)
            # self.display1(img_seg[0], data_samples[0].seg_map_path.split('\\')[-1])
            feat_from, feat_to, feat_seg, stu_from, stu_to, feat_fp_from, feat_fp_to = self.extract_feat(inputs)
            pseudo_label_cd, pseudo_from, pseudo_to, mask_from, mask_to, mask_cd = self.get_pseudo([feat_from, feat_to], data_samples)
            # x.append(x_seg)
            # print(len(x))
            losses = dict()

            loss_decode = self._decode_head_forward_train(feat_seg, 
                                                          stu_from, stu_to, 
                                                          feat_from, feat_to,
                                                          pseudo_from, pseudo_to, 
                                                          mask_from, mask_to, 
                                                          feat_fp_from, feat_fp_to, 
                                                          data_samples)
            losses.update(loss_decode)

            
            # if losses['decode.loss_stu'] > 0.1:
            #     loss_cd_decode = self._cd_decode_head_forward_train([feat_from, feat_to], pseudo_label_cd, mask_cd, data_samples)
            #     losses.update(loss_cd_decode)
            # loss_cd_decode = self._cd_decode_head_forward_train([feat_from, feat_to], pseudo_label_cd, mask_cd, data_samples)
            # losses.update(loss_cd_decode)
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train([feat_from, feat_to, feat_seg], data_samples)
                losses.update(loss_aux)

            return losses
    
    def _decode_head_forward_train(self, 
                                   inputs: List[Tensor], 
                                   stu_from: List[Tensor], 
                                   stu_to: List[Tensor],
                                   feat_from, feat_to,
                                   pseudo_from, pseudo_to,
                                   mask_from, mask_to,
                                   feat_fp_from, feat_fp_to,
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        
        loss_decode = self.decode_student.loss(inputs, data_samples,
                                            self.train_cfg)
        loss_stu = self.decode_teacher.loss_stu(stu_from, stu_to, 
                                                feat_from, feat_to,
                                                pseudo_from, pseudo_to, 
                                                mask_from, mask_to, 
                                                feat_fp_from, feat_fp_to)
        loss_decode.update(loss_stu)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    @torch.no_grad()
    def get_pseudo(self, inputs, data_samples: SampleList):
        feat_from, feat_to = inputs
        seg_logits_from = self.decode_teacher.predict(feat_from, data_samples,
                                              self.test_cfg)
        
        seg_logits_to = self.decode_teacher.predict(feat_to, data_samples,
                                              self.test_cfg)
        pseudo_label_from = torch.softmax(seg_logits_from.detach(), dim=1)
        pseudo_label_to = torch.softmax(seg_logits_to.detach(), dim=1)

        max_label_cd, pseudo_label_cd = torch.max(torch.abs(pseudo_label_to - pseudo_label_from), dim=1)
        max_label_from, pseudo_label_from = torch.max(pseudo_label_from, dim=1)
        max_label_to, pseudo_label_to = torch.max(pseudo_label_to, dim=1)
        pseudo_label = torch.logical_xor(pseudo_label_from, pseudo_label_to).int() #t1，t2时序图异或得到变化伪标签
        mask_from = max_label_from.ge(0.95).float()
        mask_to = max_label_to.ge(0.95).float()
        mask_cd = max_label_cd.ge(0.5).int()


        # self.display1(pseudo_label_from[0], data_samples[0].seg_map_path.split('\\')[-1])
        # self.display1(pseudo_label_to[0], data_samples[0].seg_map_path.split('\\')[-1])
        # self.display1(mask_cd[0], data_samples[0].seg_map_path.split('\\')[-1])
        # print(torch.unique(max_label_cd))
        # pseudo_label = (pseudo_label == pseudo_label_cd).int()
        return mask_cd, pseudo_label_from, pseudo_label_to, mask_from, mask_to, None

