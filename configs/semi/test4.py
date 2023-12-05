_base_ = [
    '../_base_/models/changer_mit-b0.py', '../_base_/datasets/levir_cd_semi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py']
crop_size = (512, 512)
# data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# checkpoint_file = '/root/autodl-tmp/logs/test/2/iter_40000.pth'
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    # size=crop_size,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
norm_cfg = dict(type='BN', requires_grad=True)
# checkpoint_file = '/root/autodl-tmp/pretrain/iter_160000.pth'
model = dict(
    type='SLLEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2))
    ),
    neck=dict(
        type='NL_FPN',
        in_dim=768,
        reduction=True),
    # decode_head=dict(
    #     type='SemiHead',
    #     in_channels=[96, 192, 384, 768], 
    #     feature_strides=[4, 8, 16, 32],
    #     num_classes=9, 
    #     ignore_index = 255,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    visualization=dict(type='CDVisualizationHook'))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict( 
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
custom_hooks = [dict(type='MeanTeacherHook')]
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
