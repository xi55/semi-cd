# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basecddataset import _BaseCDDataset
from .basedataset import _BaseDataset


@DATASETS.register_module()
class LEVIRCDDataset(_BaseCDDataset):
    """ISPRS dataset.

    In segmentation map annotation for ISPRS, 0 is to ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 format_seg_map='to_binary',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            format_seg_map=format_seg_map,
            **kwargs)
