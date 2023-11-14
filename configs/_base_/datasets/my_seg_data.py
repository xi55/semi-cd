dataset_type = 'my_seg_Dataset'
data_root = 'E:/changeDectect/train_with_seg/'

crop_size = (512, 512)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    # dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    # dict(
    #     type='MultiImgRandomResize',
    #     scale=(512, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True
    # ),
    # dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    # dict(type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    # dict(type='MultiImgResize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        # type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
            #     for r in img_ratios
            # ],
            # [
            #     dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
            #     dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            # ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train/label',
            img_path_from='train/A', 
            img_path_to='train/B',
            img_seg='seg/train/seg_imgs',
            img_seg_label='seg/train/seg_labels'
            ),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A',
            img_path_to='val/B',
            img_seg='seg/test/seg_imgs',
            img_seg_label='seg/test/seg_labels'
            ),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A',
            img_path_to='val/B',
            img_seg='seg/test/seg_imgs',
            img_seg_label='seg/test/seg_labels'
            ),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mFscore', 'mIoU'])