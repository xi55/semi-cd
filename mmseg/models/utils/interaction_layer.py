# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmseg.models.builder import ITERACTION_LAYERS
from mmseg.registry import MODELS

@ITERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2
    

@ITERACTION_LAYERS.register_module()
class MHSA_AD(BaseModule):
    def __init__(self, 
                 dim, 
                 num_heads,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'), 
                 norm_cfg=dict(type='LN'), 
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.mlp_q = nn.Linear(self.dim, self.dim)
        # self.mlp_k = nn.Linear(self.dim, self.dim)
        # self.mlp_v = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(attn_drop_rate)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

        self.ffn = FFN(
            embed_dims=dim,
            feedforward_channels=dim * 2,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

        self.scale = dim ** -0.5
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        N = H * W
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1)
        x2 = x2.reshape(B, C, -1).permute(0, 2, 1)

        # x1 = self.norm1(x1)
        # x2 = self.norm1(x2)

        Q_t1 = x1
        K_t2 = x2
        V_x1 = x1
        V_x2 = x2

        Q_t1 = Q_t1 * self.scale

        attn = Q_t1 @ K_t2.transpose(-2, -1)
        attn = 1 - self.softmax(attn)

        change_attn1 = (attn @ V_x1).transpose(1, 2).reshape(B, N, C)
        change_attn1 = self.norm2(change_attn1)
        x1 = self.ffn(change_attn1, identity=x1).permute(0, 2, 1).reshape(B, C, H, W)

        change_attn2 = (attn @ V_x2).transpose(1, 2).reshape(B, N, C)
        change_attn2 = self.norm2(change_attn2)
        x2 = self.ffn(change_attn2, identity=x2).permute(0, 2, 1).reshape(B, C, H, W)

        return x1, x2

@ITERACTION_LAYERS.register_module()
class TwoIdentity(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        return x1, x2
    







        
