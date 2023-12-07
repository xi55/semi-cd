import os
import os.path as osp

import cv2
import numpy as np
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import LocalVisBackend, force_init_env


@VISBACKENDS.register_module()
class CDLocalVisBackend(LocalVisBackend):

    @force_init_env
    def add_image(self,
                  label_name: str,
                  unlabel_name: str,
                  image: np.array,
                  semi_img: np.array = None,
                  image_strong: np.array = None,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        assert image.dtype == np.uint8

        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_label_name = f'{label_name}.png'
        save_unlabel_name = f'{unlabel_name}.png'

        if semi_img is not None:
            assert semi_img.dtype == np.uint8
            drawn_image_weak = cv2.cvtColor(semi_img, cv2.COLOR_RGB2BGR)
            for sub_dir in ['label_cd', 'unlabel_cd']:
                os.makedirs(osp.join(self._img_save_dir, sub_dir), exist_ok=True)

            cv2.imwrite(osp.join(self._img_save_dir, 'label_cd', save_label_name), drawn_image)
            cv2.imwrite(osp.join(self._img_save_dir, 'unlabel_cd', save_unlabel_name), semi_img)
        else:       
            cv2.imwrite(osp.join(self._img_save_dir, save_label_name), drawn_image)