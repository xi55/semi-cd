_base_ = [
    '../_base_/models/fpn_swin.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_data.py'
]
checkpoint_file = 'D:/git/mmseg/mmsegmentation/logs/test/4/iter_160000.pth'
detector = _base_.model
detector.data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size_divisor=32,
    bgr_to_rgb=False)
detector.backbone = dict(
    type='SwinTransformer',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    use_abs_pos_embed=False,
    drop_path_rate=0.3,
    patch_norm=True)
detector.decode_head=dict(
    type='FPNHead',
    in_channels=[96, 192, 384, 768],
    feature_strides=[4, 8, 16, 32],
    in_index=[0, 1, 2, 3],
    channels=512,
    dropout_ratio=0.1,
    num_classes=9,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

model = dict(
    _delete_=True,
    type='SoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
# labeled_dataset = _base_.labeled_dataset
# unlabeled_dataset = _base_.unlabeled_dataset
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10-unlabeled.json'
# unlabeled_dataset.data_prefix = dict(img='train2017/')
# train_dataloader = dict(
#     dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=180000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='SegVisualizationHook', interval=1))
# log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
