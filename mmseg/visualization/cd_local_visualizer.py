from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.dist import master_only

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from mmseg.registry import VISUALIZERS


@VISUALIZERS.register_module()
class CDLocalVisualizer(SegLocalVisualizer):
    """Change Detection Local Visualizer. """

    @master_only
    def add_datasample(
            self,
            label_name: str,
            unlabel_name: str,
            image: np.ndarray,
            image_from_to: Sequence[np.array],
            data_batch: dict,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            image_from_to (Sequence[np.array]): The image pairs to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        exist_img_from_to = True if len(image_from_to) > 0 else False
        if exist_img_from_to:
            assert len(image_from_to) == 2, '`image_from_to` contains `from` ' \
                'and `to` images'

        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)
        # semantic_classes = self.dataset_meta.get('semantic_classes', None)
        # semantic_palette = self.dataset_meta.get('semantic_palette', None)
        semantic_classes = ('background', 'changed')
        semantic_palette = [[0, 0, 0], [255, 255, 255]]
        gt_img_data = None
        gt_img_data_from = None
        gt_img_data_to = None
        pred_img_data = None
        pred_img_data_from = None
        pred_img_data_to = None
        
        drawn_img_from = None
        drawn_img_to = None

        # print(data_sample.keys())
        # print(draw_gt)
        # print(data_batch.keys())
        # print(data_batch['data_samples'][0].keys())

        if draw_gt and data_batch['data_samples'][0] is not None and 'gt_sem_seg' in data_batch['data_samples'][0]:
            gt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            gt_img_l_data = self._draw_sem_seg(gt_img_data,
                                             data_batch['data_samples'][0].gt_sem_seg, classes,
                                             palette)
        if draw_gt and data_sample is not None and 'i_cd_pred' in data_sample:
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            img_data = image
            pred_img_l_data = self._draw_sem_seg(img_data,
                                             data_sample.i_cd_pred, classes,
                                             palette)

        if (draw_pred and data_sample is not None and 'w_cd_pred' in data_sample):
            pseudo_label_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_w_data = self._draw_sem_seg(pseudo_label_data,
                                               data_sample.w_cd_pred,
                                               classes, palette)
        
        if (draw_pred and data_sample is not None and 's_cd_pred' in data_sample):
            pseudo_label_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_s_data = self._draw_sem_seg(pseudo_label_data,
                                               data_sample.s_cd_pred,
                                               classes, palette)
            
        # if (draw_pred and data_sample is not None and 'i_seg_stu_from_pred' in data_sample \
        #     and 'i_seg_stu_to_pred' in data_sample):
        #     if exist_img_from_to:
        #         pred_img_data_from = image_from_to[0]
        #         pred_img_data_to = image_from_to[1]
        #     else:
        #         pred_img_data_from = np.zeros_like(image)
        #         pred_img_data_to = np.zeros_like(image)
        #     assert classes is not None, 'class information is ' \
        #                                 'not provided when ' \
        #                                 'visualizing change ' \
        #                                 'deteaction results.'
        #     pred_img_data_from = self._draw_sem_seg(pred_img_data_from,
        #                                      data_sample.i_seg_stu_from_pred, classes, palette)
        #     pred_img_data_to = self._draw_sem_seg(pred_img_data_to,
        #                                      data_sample.i_seg_stu_to_pred, classes, palette)

        if gt_img_l_data is not None and pred_img_l_data is not None:
            drawn_img = np.concatenate((gt_img_l_data, pred_img_l_data), axis=1)
        elif gt_img_l_data is not None:
            drawn_img = gt_img_l_data
        else:
            drawn_img = pred_img_l_data

        if pred_w_data is not None and pred_s_data is not None:
            drawn_semi_img = np.concatenate((pred_w_data, pred_s_data), axis=1)
        elif pred_w_data is not None:
            drawn_semi_img = pred_w_data
        elif pred_s_data is not None:
            drawn_semi_img = pred_s_data
        else:
            drawn_semi_img = None

        # if gt_img_data_from is not None and pred_img_data_from is not None:
        #     drawn_img_weak = np.concatenate((gt_img_data_from, gt_img_data_to), axis=1)
        # elif gt_img_data_from is not None:
        #     drawn_img_weak = gt_img_data_from
        # else:
        #     drawn_img_weak = pred_img_data_from

        # if gt_img_data_to is not None and pred_img_data_to is not None:
        #     drawn_img_strong = np.concatenate((pred_img_data_from, pred_img_data_to), axis=1)
        # elif gt_img_data_to is not None:
        #     drawn_img_strong = gt_img_data_to
        # else:
        #     drawn_img_strong = pred_img_data_to

        # if show:
        #     if drawn_img_weak is not None and drawn_img_strong is not None:
        #         drawn_img_cat = np.concatenate((drawn_img, drawn_img_weak, drawn_img_strong), axis=0)
        #         self.show(drawn_img_cat, win_name=name, wait_time=wait_time)
                
        #     else:
        #         self.show(drawn_img, win_name=name, wait_time=wait_time)
        # print(name)
        # print(drawn_img.shape)
        # print(drawn_img_from.shape)
        # print(drawn_img_to.shape)
        # if out_file is not None:
        #     if drawn_img_weak is not None and drawn_img_strong is not None:
        #         drawn_img_cat = np.concatenate((drawn_img, drawn_img_weak, drawn_img_strong), axis=0)
        #         mmcv.imwrite(mmcv.bgr2rgb(drawn_img_cat), out_file)
        #     else:
        #         mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
        # else:
        self.add_image(label_name, unlabel_name, drawn_img, drawn_semi_img, None, step)

    @master_only
    def add_image(self, 
                  label_name: str,
                  unlabel_name: str,
                  image: np.ndarray,
                  semi_img: np.ndarray = None,
                  image_to: np.ndarray = None,
                  step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(label_name, unlabel_name, image, semi_img, image_to, step)  # type: ignore

    @master_only
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]

        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        self.fig_save.set_size_inches(  # type: ignore
            self.width / self.dpi, self.height / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')
