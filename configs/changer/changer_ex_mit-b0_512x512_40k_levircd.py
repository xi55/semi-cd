_base_ = [
        '../_base_/models/changer_mit-b0.py', '../_base_/datasets/levir_cd_semi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py']

crop_size = (512, 512)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    type='ChangerEncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        interaction_cfg=(
            None,
            # dict(type='MHSA_AD', dim=32, num_heads=2),
            dict(type='SpatialExchange', p=1/2),
            # dict(type='MHSA_AD', dim=64, num_heads=4),
            dict(type='ChannelExchange', p=1/2),
            # dict(type='MHSA_AD', dim=160, num_heads=10),
            dict(type='MHSA_AD', dim=256, num_heads=16))
    ),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

# train_pipeline = [
#     dict(type='MultiImgLoadImageFromFile'),
#     dict(type='MultiImgLoadAnnotations'),
#     dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
#     dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='MultiImgExchangeTime', prob=0.5),
#     dict(
#         type='MultiImgPhotoMetricDistortion',
#         brightness_delta=10,
#         contrast_range=(0.8, 1.2),
#         saturation_range=(0.8, 1.2),
#         hue_delta=10),
#     dict(type='MultiImgPackSegInputs')
# ]

# train_dataloader = dict(
#     dataset=dict(pipeline=train_pipeline))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    visualization=dict(type='CDVisualizationHook'))

val_evaluator = dict(type='mmseg.IoUMetricCD', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='mmseg.IoUMetricCD',
    iou_metrics=['mFscore', 'mIoU'])

custom_hooks = [dict(type='MeanTeacherHook')]
# val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
# test_evaluator = dict(
#     type='mmseg.IoUMetric',
#     iou_metrics=['mFscore', 'mIoU'])
train_dataloader = dict(batch_size=8)