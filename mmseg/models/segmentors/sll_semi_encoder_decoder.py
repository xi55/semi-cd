# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.utils import resize
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.registry import MODELS
import matplotlib.pyplot as plt

@MODELS.register_module()
class SllSemiEncoderDecoder(BaseSegmentor):
    """SiamEncoder Decoder change detector.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        backbone_inchannels (int): The `in_channels` for backbone network.
            Defaults: 3 for RGB image.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 cd_decode_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        # self.backbone = MODELS.build(backbone)
        self._init_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backbone_inchannels = backbone_inchannels # RGB: 3

        assert self.with_decode_head

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
    
    def _init_cd_decode_head(self, cd_decode_head:ConfigType) -> None:
        if cd_decode_head is not None:
            self.cd_decode_head = MODELS.build(cd_decode_head)

    def _init_neck(self, neck: ConfigType):
        if neck is not None:
            self.neck_student = MODELS.build(neck)
            self.neck_teacher = MODELS.build(neck)
            self.freeze(self.neck_teacher)

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        feat_from = self.backbone(img_from)
        feat_to = self.backbone(img_to)
        if self.with_neck:
            x = self.neck(feat_from, feat_to)
        else:
            raise ValueError('`NECK` is needed for `SiamEncoderDecoder`.')
        
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _cd_decode_head_forward_train(self, inputs: List[Tensor],
                                   pseudo_label_cd, mask_cd, data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # print(pseudo_label_cd.shape)
        # print(torch.unique(pseudo_label_cd))
        loss_cd = self.cd_decode_head.loss(inputs, pseudo_label_cd, mask_cd, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_cd, 'cd_decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

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
        
        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, None)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def postprocess_result(self,
                           seg_logits: Tensor,
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
        if(isinstance(seg_logits, list)):
            res = self.ssl_postprocess_result(seg_logits, data_samples)
            return res

        batch_size, C, H, W = seg_logits.shape

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
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              0.5).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
    

    def ssl_postprocess_result(self,
                           seg_logits: List[Tensor],
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
        w_logits_cd, s_logits_cd, fp_logits_cd, l_logits_cd = seg_logits
        
        batch_size, C, H, W = l_logits_cd.shape
        # print(seg_logits_seg.shape)
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
                i_cd_logits = l_logits_cd[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                w_cd_logits = w_logits_cd[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                s_cd_logits = s_logits_cd[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                # i_seg_cd = seg_logits_cd[i:i + 1, :,
                #                           padding_top:H - padding_bottom,
                #                           padding_left:W - padding_right]
                # i_seg_stu_from = seg_logits_stu_from[i:i + 1, :,
                #                           padding_top:H - padding_bottom,
                #                           padding_left:W - padding_right]
                # i_seg_stu_to = seg_logits_stu_to[i:i + 1, :,
                #                           padding_top:H - padding_bottom,
                #                           padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_cd_logits = i_cd_logits.flip(dims=(3, ))
                        w_cd_logits = w_cd_logits.flip(dims=(3, ))
                        s_cd_logits = s_cd_logits.flip(dims=(3, ))
                        # i_seg_cd = i_seg_cd.flip(dims=(3, ))
                        # i_seg_stu_from = i_seg_stu_from.flip(dims=(3, ))
                        # i_seg_stu_to = i_seg_stu_to.flip(dims=(3, ))
                    else:
                        i_cd_logits = i_cd_logits.flip(dims=(2, ))
                        w_cd_logits = w_cd_logits.flip(dims=(2, ))
                        s_cd_logits = s_cd_logits.flip(dims=(2, ))
                        # i_seg_cd = i_seg_cd.flip(dims=(2, ))
                        # i_seg_stu_from = i_seg_stu_from.flip(dims=(2, ))
                        # i_seg_stu_to = i_seg_stu_to.flip(dims=(2, ))

                # resize as original shape
                res=[]
                for i_seg in [i_cd_logits, w_cd_logits, s_cd_logits]:
                    i_seg = resize(
                        i_seg,
                        size=img_meta['ori_shape'],
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False).squeeze(0)
                    res.append(i_seg)
                i_cd_logits, w_cd_logits, s_cd_logits = res
            else:
                i_cd_logits = l_logits_cd[i]
                w_cd_logits = w_logits_cd[i]
                s_cd_logits = s_logits_cd[i]
                # i_seg_cd = seg_logits_cd[i]
                # i_seg_stu_from = seg_logits_stu_from[i]
                # i_seg_stu_to = seg_logits_stu_to[i]
            # print(torch.unique(i_seg_logits))
            if C > 1:
                # print(i_seg_from.shape)
                i_cd_pred = i_cd_logits.argmax(dim=0, keepdim=True)
                w_cd_pred = w_cd_logits.argmax(dim=0, keepdim=True)
                s_cd_pred = s_cd_logits.argmax(dim=0, keepdim=True)
                # i_seg_stu_from_pred = i_seg_stu_from.argmax(dim=0, keepdim=True)
                # i_seg_stu_to_pred = i_seg_stu_to.argmax(dim=0, keepdim=True)
                # print(torch.unique(i_seg_from_pred))
            else:
                # print(1)
                i_cd_logits = i_cd_logits.sigmoid()
                w_cd_logits = w_cd_logits.sigmoid()
                s_cd_logits = s_cd_logits.sigmoid()
                # i_seg_stu_from = i_seg_stu_from.sigmoid()
                # i_seg_stu_to = i_seg_stu_to.sigmoid()
                
                
                i_cd_pred = (i_cd_logits >
                              0.5).to(i_cd_logits)
                w_cd_pred = (w_cd_logits >
                              0.5).to(w_cd_logits)
                s_cd_pred = (s_cd_logits >
                              0.5).to(s_cd_logits)
                # i_seg_stu_from_pred = (i_seg_stu_from >
                #               0.5).to(i_seg_stu_from)
                # i_seg_stu_to_pred = (i_seg_stu_to >
                #               0.5).to(i_seg_stu_to)
                
            # i_seg_cd_pred = i_seg_cd.sigmoid()

            # i_seg_cd_pred[i_seg_cd_pred >= 0.5] = 1
            # i_seg_cd_pred[i_seg_cd_pred < 0.5] = 0
            # print(i_cd_pred.shape)
            # self.display1(w_cd_pred, "1")
            data_samples[i].set_data({
                'i_cd_pred':
                PixelData(**{'data': i_cd_pred}),
                'w_cd_pred':
                PixelData(**{'data': w_cd_pred}),
                's_cd_pred':
                PixelData(**{'data': s_cd_pred}),
                # 'i_seg_to_pred':
                # PixelData(**{'data': i_seg_to_pred}),
                # 'i_seg_cd_pred':
                # PixelData(**{'data': i_seg_cd_pred}),
                # 'i_seg_stu_from_pred':
                # PixelData(**{'data': i_seg_stu_from_pred}),
                # 'i_seg_stu_to_pred':
                # PixelData(**{'data': i_seg_stu_to_pred})
            })
        # print(torch.unique(data_samples[0].i_seg_from_pred.data))
        return data_samples

    def display1(self, affinity, s):
        affinity = affinity.permute(1, 2, 0)
        affinity = affinity.cpu().numpy()
        plt.imshow(affinity)
        plt.title(s)
        plt.colorbar()
        plt.show()