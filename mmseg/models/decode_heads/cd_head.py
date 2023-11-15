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
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)
        self.conv_out_class = torch.nn.Conv2d(channel_list[3],self.out_channels, kernel_size=1,stride=1,padding=0)



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


    def loss(self, inputs: Tuple[Tensor], pseudo_label: Tensor, data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
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
        losses = self.loss_by_feat(seg_logits, pseudo_label, data_samples)
        return losses

    def loss_by_feat(self, seg_logits: Tensor,
                     pseudo_label: Tensor, data_samples: SampleList) -> dict:

        loss = dict()

        pseudo_label_size = pseudo_label.shape[2:]

        seg_logits = resize(
            input=seg_logits,
            size=pseudo_label_size,
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, pseudo_label)
        else:
            seg_weight = None
        pseudo_label = pseudo_label.squeeze(1)

        # vis = pseudo_label[0]
        # vis[vis == 255.0] == 2
        # self.display1(vis, data_samples[0].seg_map_path.split('\\')[-1])
        # print(seg_logits[0])
        # print(pseudo_label[0])
        # print(torch.unique(seg_logits))
        # print(torch.unique(pseudo_label))
        loss['loss_cd_seg'] = self.loss_decode(
                    seg_logits,
                    pseudo_label.long(),
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # print(loss['loss_cd_seg'])
        # seg_label = seg_label.squeeze(1)
        loss['acc_seg'] = accuracy(
            seg_logits, pseudo_label, ignore_index=self.ignore_index)
        return loss
    

    def predict_by_feat(self, seg_logits: Tensor,
                        pseudo_label: Tensor) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = resize(
                input=seg_logits,
                size=pseudo_label[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners)
        return seg_logits
        

