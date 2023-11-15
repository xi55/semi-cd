# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional
from mmseg.utils import (ConfigType, SampleList, add_prefix)
import torch
from torch import Tensor
import torch.nn as nn
from mmseg.registry import MODELS
from .semi_encoder_decoder import SiamEncoderDecoder
import matplotlib.pyplot as plt

@MODELS.register_module()
class SLLEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self, backbone, decode_head, pretrained: Optional[str] = None, *args, **kwargs):
        super().__init__(backbone = backbone, decode_head=decode_head, *args, **kwargs)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone_student = MODELS.build(backbone)
        self.backbone_teacher = MODELS.build(backbone)
        self._init_decode_head(decode_head)
        self.freeze(self.backbone_teacher)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
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
    
    def extract_feat(self, inputs) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        if isinstance(inputs, Tensor):
            img_from, img_to, img_seg = torch.split(inputs, self.backbone_inchannels, dim=1)
        elif isinstance(inputs, list):
            img_from, img_to, img_seg = inputs

        feat_from = self.backbone_teacher(img_from)
        feat_to = self.backbone_teacher(img_to)

        # stu_from = self.backbone_student(img_from)
        # stu_to = self.backbone_student(img_to)

        feat_seg = self.backbone_student(img_seg)

        if self.with_neck:
            feat_from[3] = self.neck(feat_from[3])
            feat_to[3] = self.neck(feat_to[3])
            # feat_seg[3] = self.neck(feat_seg[3])

            # stu_from[3] = self.neck(stu_from[3])
            # stu_to[3] = self.neck(stu_to[3])

        # x.append(x_seg)
        return feat_from, feat_to, feat_seg
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feat_from, feat_to, feat_seg  = self.extract_feat(inputs)
        seg_logits_from = self.decode_teacher.predict(feat_from, batch_img_metas,
                                              self.test_cfg)
        
        seg_logits_to = self.decode_teacher.predict(feat_to, batch_img_metas,
                                              self.test_cfg)

        seg_logits_seg = self.decode_student.predict(feat_seg, batch_img_metas,
                                                    self.test_cfg)
        
        seg_logits_cd = self.cd_decode_head.predict([feat_from, feat_to], batch_img_metas,
                                                    self.test_cfg)
        # print(batch_img_metas[0].keys())
        # self.display1(seg_logits_cd[0], batch_img_metas[0]['seg_map_path'].split('\\')[-1])
        return [seg_logits_from, seg_logits_to, seg_logits_seg, seg_logits_cd]

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
            # print()
            # self.display1(img_seg[0], data_samples[0].seg_map_path.split('\\')[-1])
            feat_from, feat_to, feat_seg = self.extract_feat(inputs)
            
            # x.append(x_seg)
            # print(len(x))
            losses = dict()
            loss_decode = self._decode_head_forward_train(feat_seg, data_samples)
            losses.update(loss_decode)

            pseudo_label = self.get_pseudo(inputs, data_samples)

            loss_cd_decode = self._cd_decode_head_forward_train([feat_from, feat_to], pseudo_label, data_samples)
            losses.update(loss_cd_decode)
            
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train([feat_from, feat_to, feat_seg], data_samples)
                losses.update(loss_aux)

            return losses
    

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        
        loss_decode = self.decode_student.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    @torch.no_grad()
    def get_pseudo(self, inputs, data_samples):
        pseudo_pred = self.predict(inputs, data_samples) #语义分割预测输出
        pseudo_from = self._stack_batch_from(pseudo_pred) #t1时刻时序图伪标签
        pseudo_to = self._stack_batch_to(pseudo_pred) #t2时刻时序图伪标签
        # pseudo_label = torch.logical_xor(pseudo_from.int(), pseudo_to.int()).int() #t1，t2时序图异或得到变化伪标签
        pseudo_label = torch.tensor((pseudo_from != pseudo_to), dtype=torch.int32, device='cuda:0', requires_grad=False)
        # self.display1(pseudo_from[0], data_samples[0].seg_map_path.split('\\')[-1])
        # self.display1(pseudo_label[0], data_samples[0].seg_map_path.split('\\')[-1])
        # self.display1(pseudo_pred[0].i_seg_cd_pred.data, data_samples[0].seg_map_path.split('\\')[-1])
        # print('pseudo_pred: ')
        # print(pseudo_pred[0].i_seg_cd_pred.data)
        return pseudo_label

