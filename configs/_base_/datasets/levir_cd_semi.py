
dataset_type = 'LEVIRCDDataset'
data_root = '/root/autodl-tmp/LEVIR_CD_SEMI'

crop_size = (512, 512)

color_space = [
    # [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgPhotoMetricDistortion'),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            # dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]
        ),
    
    # dict(
    #     type='MultiImgRandomResize',
    #     scale=(512, 512),
    #     ratio_range=(0.5, 1.5),
    #     keep_ratio=True
    # ),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    # dict(type='MultiImgPhotoMetricDistortion'),
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
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            label_path='train/label',
            img_path_label_from='train/train_l/A', 
            img_path_label_to='train/train_l/B',
            img_path_unlabel_from='train/train_u/A', 
            img_path_unlabel_to='train/train_u/B'
            ),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            label_path='val/label',
            label_u_path='val/label_u',
            img_path_label_from='val/val_l/A', 
            img_path_label_to='val/val_l/B',
            img_path_unlabel_from='val/val_u/A', 
            img_path_unlabel_to='val/val_u/B'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            label_path='val/label',
            label_u_path='val/label_u',
            img_path_label_from='val/val_l/A', 
            img_path_label_to='val/val_l/B',
            img_path_unlabel_from='val/val_u/A', 
            img_path_unlabel_to='val/val_u/B'),
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(
    type='mmseg.IoUMetric',
    iou_metrics=['mFscore', 'mIoU'])