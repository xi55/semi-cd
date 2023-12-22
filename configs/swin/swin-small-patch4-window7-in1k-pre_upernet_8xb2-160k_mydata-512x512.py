_base_ = [
    './swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_mydata-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=8, ignore_index = 255),
    auxiliary_head=None
    )

train_dataloader = dict(batch_size=8)
