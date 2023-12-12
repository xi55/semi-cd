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
class ChangerEncoderDecoder(SllSemiEncoderDecoder):
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
        img_s_from=None
        img_s_to=None
        img_l_from, img_l_to, img_u_from, img_u_to, img_s_from, img_s_to = torch.split(inputs, self.backbone_inchannels, dim=1)

        feat_w = self.backbone_teacher(img_u_from, img_u_to)

        feat_fp = [nn.Dropout2d(0.5)(f) for f in feat_w]

        feat_s = self.backbone_teacher(img_s_from, img_s_to)

        feat_l = self.backbone_student(img_l_from, img_l_to)

        # if self.with_neck:
        #     feat_w_from[3] = self.neck_teacher(feat_w_from[3])
        #     feat_w_to[3] = self.neck_teacher(feat_w_to[3])
            
        #     feat_fp_from[3] = self.neck_teacher(feat_fp_from[3])
        #     feat_fp_to[3] = self.neck_teacher(feat_fp_to[3])

        #     feat_s_from[3] = self.neck_teacher(feat_s_from[3])
        #     feat_s_to[3] = self.neck_teacher(feat_s_to[3])

        #     feat_l_from[3] = self.neck_student(feat_l_from[3])
        #     feat_l_to[3] = self.neck_student(feat_l_to[3])

        # x.append(x_seg)
        return feat_w, feat_fp, feat_s, feat_l
    
    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feat_w, feat_fp, feat_s, feat_l = self.extract_feat(inputs)
        w_logits_cd = self.decode_teacher.predict(feat_w, batch_img_metas,
                                              self.test_cfg)

        fp_logits_cd = self.decode_teacher.predict(feat_fp, batch_img_metas,
                                                    self.test_cfg)
        
        s_logits_cd = self.decode_teacher.predict(feat_s, batch_img_metas,
                                                    self.test_cfg)

        l_logits_cd = self.decode_student.predict(feat_l, batch_img_metas,
                                                    self.test_cfg)
        # print(batch_img_metas[0].keys())
        # self.display1(seg_logits_cd[0], batch_img_metas[0]['seg_map_path'].split('\\')[-1])
        
        return [w_logits_cd, s_logits_cd, fp_logits_cd, l_logits_cd]

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
        loss_stu = self.decode_teacher.loss_stu(feat_w,
                                                feat_fp,
                                                feat_s,
                                                pseudo_label, mask)
        loss_decode.update(loss_stu)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    @torch.no_grad()
    def get_pseudo(self, inputs, data_samples: SampleList):
        w_logits_cd = self.decode_teacher.predict(inputs, data_samples, self.test_cfg)
        # print(w_logits_cd.shape)
        # pseudo_label_cd = w_logits_cd.sigmoid()
        # pseudo_label = (pseudo_label_cd > 0.5).to(pseudo_label_cd)

        pseudo_label_cd = torch.softmax(w_logits_cd.detach(), dim=1)
        pseudo_mask, pseudo_label = torch.max(pseudo_label_cd, dim=1)
        mask = pseudo_mask.ge(0.95).float()
        # print(torch.unique(pseudo_label_cd))
        # print(torch.unique(pseudo_label))
        # print(torch.unique(pseudo_mask))
        # print(torch.unique(mask))
        return pseudo_label, mask

