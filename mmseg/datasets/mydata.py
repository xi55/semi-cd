# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from .basesemidataset import BaseSemiDataset


@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Water', 'Transport', 'Building', 'Arableland', 'Grassland', 'Woodland', 'land', 'other'),
        palette=[[0, 0, 255], [211, 211, 211], [255, 0, 0], [255, 255, 0], [0, 255, 0], [34, 139, 34], [139, 117, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
