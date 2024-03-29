import torch.nn as nn
import torch
from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS
import functools
from torch import Tensor
from mmseg.utils import SampleList
from mmseg.models.utils import resize
import numpy as np
import cv2
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
import matplotlib.pyplot as plt

bn_mom = 0.0003
class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)

class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample = False):
        super(cat,self).__init__() ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv2d=torch.nn.Sequential(
            torch.nn.Conv2d(in_chn_high + in_chn_low, out_chn, kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )
    
    def forward(self,x,y):
        # import ipdb
        # ipdb.set_trace()
        if self.do_upsample:
            x = self.upsample(x)

        x = torch.cat((x,y),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)
    
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                torch.nn.BatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
    
@MODELS.register_module()  
class SSL_CD_Head(BaseDecodeHead):
    def __init__(self, pool_list = [True, True, True, True], **kwargs):
        super(SSL_CD_Head, self).__init__(input_transform='multiple_select', **kwargs)
        channel_list = self.in_channels
        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        self.catc3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        self.upsample_x1=nn.Sequential(
                        nn.Conv2d(channel_list[3],channel_list[3],kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(channel_list[3], momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,2,kernel_size=3,stride=1,padding=1)
        # self.conv_out_class = torch.nn.Conv2d(8,9,kernel_size=3,stride=1,padding=1)



    def forward(self, x):
        x1, x2 = x[0], x[1]

        c = self.df4(x1[3], x2[3])

        y1 = self.decoder3(x1[3], x1[2])
        y2 = self.decoder3(x2[3], x2[2])
        
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, x1[1])
        y2 = self.decoder2(y2, x2[1])
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, x1[0])
        y2 = self.decoder1(y2, x2[0])
        c = self.catc1(c, self.df1(y1, y2))

        y = self.conv_out(self.upsample_x2(c))
        return y


    def display1(self, affinity, s):
        affinity = affinity.unsqueeze(2)
        affinity = affinity.cpu().numpy()
        plt.imshow(affinity)
        plt.title(s)
        plt.colorbar()
        plt.show()

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, inputs: Tuple[Tensor], data_samples, train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, data_samples)
        return losses

    def loss_stu(self, 
                 feat_w_from, feat_w_to,
                 feat_fp_from, feat_fp_to,
                 feat_s_from, feat_s_to,
                 pseudo_label, mask):
        cd_s_logits = self.forward([feat_s_from, feat_s_to])

        cd_fp_logits = self.forward([feat_fp_from, feat_fp_to])
        
        loss = dict()
        
        u_preds = [resize(input=u_pred, size=pseudo_label.shape[1:], 
                          mode='bilinear', align_corners=self.align_corners) 
                          for u_pred in [cd_s_logits, cd_fp_logits]]
        cd_s_logits, cd_fp_logits = u_preds


        if self.sampler is not None:
            cd_weight = self.sampler.sample(cd_s_logits, pseudo_label)
        else:
            cd_weight = None
        # pseudo_label = pseudo_label.squeeze(1)
        # print(pseudo_label.shape)
        # print(cd_s_logits.shape)
        # print(pseudo_label.shape)

        loss_strong = self.loss_decode(
            cd_s_logits,
            pseudo_label,
            weight=mask,
            ignore_index=self.ignore_index)
        
        loss_fp = self.loss_decode(
            cd_fp_logits,
            pseudo_label,
            weight=mask,
            ignore_index=self.ignore_index)
        
        loss['loss_s1'] = 0.5 * loss_strong
        loss['loss_fp'] = 0.5 * loss_fp
        
        return loss

    def loss_by_feat(self, cd_logits: Tensor,
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
        cd_label = self._stack_batch_gt(batch_data_samples)
        # print(torch.unique(cd_label))
        # print(cd_logits.shape)
        loss = dict()
        cd_logits = resize(
            input=cd_logits,
            size=cd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            cd_weight = self.sampler.sample(cd_logits, cd_label)
        else:
            cd_weight = None
        cd_label = cd_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        # print(cd_logits.shape)
        # print(cd_label.shape)
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    cd_logits,
                    cd_label,
                    weight=cd_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    cd_logits,
                    cd_label,
                    weight=cd_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            cd_logits, cd_label, ignore_index=self.ignore_index)
        return loss