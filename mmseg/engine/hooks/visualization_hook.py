# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
import numpy as np
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer, CDLocalVisualizer


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = data_batch['data_samples'][0].img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)



@HOOKS.register_module()
class CDVisualizationHook(SegVisualizationHook):
    """Change Detection Visualization Hook. Used to visualize validation and
    testing process prediction results. 

    Args:
        img_shape (tuple): if img_shape is given and `draw_on_from_to_img` is
            False, the original images will not be read.
        draw_on_from_to_img (bool): whether to draw semantic prediction results
            on the original images. If it is False, it means that drawing on
            the black board. Defaults to False.
    
    """
    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = True,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self.img_shape = img_shape
        self.draw_on_from_to_img = draw_on_from_to_img
        if self.draw_on_from_to_img:
            warnings.warn('`draw_on_from_to_img` works only in '
                          'semantic change detection.')
        self._visualizer: CDLocalVisualizer = \
            CDLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        
        self.backend_args = backend_args.copy() if backend_args else None
        
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return
        if self.every_n_inner_iters(batch_idx, self.interval):

            for output in outputs:
                img_path = data_batch['data_samples'][0].img_path[0]
                img_from_to = []
                window_name = osp.basename(img_path).split('.')[0]
                if self.img_shape is not None:
                    assert len(self.img_shape) == 3, \
                        '`img_shape` should be (H, W, C)'
                else:
                    img_bytes = fileio.get(
                        img_path, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    self.img_shape = img.shape

                if self.draw_on_from_to_img:
                    # for semantic change detection
                    for _img_path in data_batch['data_samples'][0].img_path:
                        _img_bytes = fileio.get(
                            _img_path, backend_args=self.backend_args)
                        _img = mmcv.imfrombytes(_img_bytes, channel_order='rgb')
                        img_from_to.append(_img)
                        
                img = np.zeros(self.img_shape)
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    img_from_to,
                    data_batch=data_batch,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter,
                    draw_gt=True)
