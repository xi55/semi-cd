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
                  name: str,
                  image: np.array,
                  image_weak: np.array = None,
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
        save_file_name = f'{name}.png'

        if image_weak is not None and image_strong is not None:
            assert image_weak.dtype == np.uint8 and image_strong.dtype == np.uint8
            drawn_image_weak = cv2.cvtColor(image_weak, cv2.COLOR_RGB2BGR)
            drawn_image_strong = cv2.cvtColor(image_strong, cv2.COLOR_RGB2BGR)
            for sub_dir in ['seg', 'weak', 'strong']:
                os.makedirs(osp.join(self._img_save_dir, sub_dir), exist_ok=True)

            cv2.imwrite(osp.join(self._img_save_dir, 'seg', save_file_name), drawn_image)
            cv2.imwrite(osp.join(self._img_save_dir, 'weak', save_file_name), drawn_image_weak)
            cv2.imwrite(osp.join(self._img_save_dir, 'strong', save_file_name), drawn_image_strong)
        else:       
            cv2.imwrite(osp.join(self._img_save_dir, save_file_name), drawn_image)