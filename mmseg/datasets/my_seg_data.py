# Copyright (c) Open-CD. All rights reserved.
from mmseg.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class my_seg_Dataset(_BaseCDDataset):
    METAINFO = dict(
        classes=('BackGround', 'Water', 'Transport', 'Building', 'Arableland', 'Grassland', 'Woodland', 'land', 'other'),
        palette=[[0, 0, 0], [0, 0, 255], [211, 211, 211], [255, 0, 0], [255, 255, 0], [0, 255, 0], [34, 139, 34], [139, 117, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 data_prefix: dict = dict(img_path='', seg_map_path='', img_seg='', img_seg_label=''),
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            data_prefix = data_prefix,
            **kwargs)
