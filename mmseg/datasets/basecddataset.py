# Copyright (c) Open-CD. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS


@DATASETS.register_module()
class _BaseCDDataset(BaseDataset):
    """用于变化检测的自定义数据集。文件结构示例如下所示：
.. code-block:: none
    ├── data
    │   ├── my_dataset
    │   │   ├── train
    │   │   │   ├── img_path_from/xxx{img_suffix}
    │   │   │   ├── img_path_to/xxx{img_suffix}
    │   │   │   ├── seg_map_path/xxx{img_suffix}
    │   │   ├── val
    │   │   │   ├── img_path_from/xxx{seg_map_suffix}
    │   │   │   ├── img_path_to/xxx{seg_map_suffix}
    │   │   │   ├── seg_map_path/xxx{seg_map_suffix}

CustomDataset的imgs/gt_semantic_seg成对,除了后缀之外都应该相同。有效的img/gt_semantic_seg文件名对应为``xxx{img_suffix}``和``xxx{seg_map_suffix}``
(后缀也包括在内)。如果提供了split参数,那么``xxx``将在txt文件中指定。否则,将加载``img_path_x/``和``seg_map_path``中的所有文件。
更多详细信息请参考``docs/en/tutorials/new_dataset.md``。

参数：
    ann_file (str): 注释文件路径。默认为空字符串。
    metainfo (dict, 可选): 数据集的元信息,如指定要加载的类别。默认为None。
    data_root (str, 可选): ``data_prefix``和``ann_file``的根目录。默认为None。
    data_prefix (dict, 可选): 训练数据的前缀。默认为dict(img_path=None, seg_map_path=None)。
    img_suffix (str): 图像的后缀。默认为'.jpg'。
    seg_map_suffix (str): 分割地图的后缀。默认为'.png'。
    format_seg_map (str): 如果`format_seg_map`='to_binary',二进制变化检测标签将被格式化为0(<128)或1(>=128)。默认为None。
    filter_cfg (dict, 可选): 用于筛选数据的配置。默认为None。
    indices (int或Sequence[int], 可选): 支持使用注释文件中的前几个数据来便于在较小的数据集上进行训练/测试。默认为None,表示使用所有``data_infos``。
    serialize_data (bool, 可选): 是否使用序列化对象来保存内存,启用后,数据加载器工作进程可以使用主进程的共享内存,而不是制作副本。默认为True。
    pipeline (list, 可选): 数据处理流程。默认为空列表。
    test_mode (bool, 可选): ``test_mode=True``表示处于测试阶段。默认为False。
    lazy_init (bool, 可选): 是否在实例化期间加载注释。在某些情况下,例如可视化,只需要数据集的元信息,无需加载注释文件。通过设置``lazy_init=True``,Basedataset可以跳过加载注释以节省时间。默认为False。
    max_refetch (int, 可选): 如果``Basedataset.prepare_data``获取了空图像,获取有效图像的最大额外循环次数。默认为1000。
    ignore_index (int): 要忽略的标签索引。默认为255。
    reduce_zero_label (bool): 是否将标签零标记为忽略。默认为False。
    backend_args (dict, 可选): 用于实例化文件后端的参数。有关详细信息,请参阅https://mmengine.readthedocs.io/en/latest/api/fileio.htm。默认为None。
    注意:需要mmcv>=2.0.0rc4和mmengine>=0.2.0。
"""

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 format_seg_map=None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.format_seg_map = format_seg_map
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

        img_dir_from = self.data_prefix.get('img_path_from', None)
        img_dir_to = self.data_prefix.get('img_path_to', None)
        img_seg = self.data_prefix.get("img_seg", None)
        img_seg_label = self.data_prefix.get("img_seg_label", None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        if osp.isfile(self.ann_file):

            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(img_path=\
                                 [osp.join(img_dir_from, img_name + self.img_suffix), \
                                  osp.join(img_dir_to, img_name + self.img_suffix)])
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:

            file_list_from = fileio.list_dir_or_file(
                    dir_path=img_dir_from,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args)
            file_list_to = fileio.list_dir_or_file(
                    dir_path=img_dir_to,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args)

            assert sorted(list(file_list_from)) == sorted(list(file_list_to)), \
                'The images in `img_path_from` and `img_path_to` are not ' \
                    'one-to-one correspondence'

            for img in fileio.list_dir_or_file(
                    dir_path=img_dir_from,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=\
                                 [osp.join(img_dir_from, img), \
                                  osp.join(img_dir_to, img)])
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if img_seg is not None:
                    data_info['img_seg'] = osp.join(img_seg, img)
                    data_info['img_seg_label'] = osp.join(img_seg_label, seg_map)
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
