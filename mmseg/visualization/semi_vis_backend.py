import os
import os.path as osp

import cv2
import numpy as np
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import LocalVisBackend, force_init_env


@VISBACKENDS.register_module()
class SemiLocalVisBackend(LocalVisBackend):

    @force_init_env
    def add_image(self,
                  name_l: str,
                  name_u: str,
                  image_l: np.array,
                  image_u: np.array = None,
                  step: int = 1,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name_l = f'{name_l}.png'
        save_file_name_u = f'{name_u}.png'
        if image_l is not None and image_u is not None:
            assert image_l.dtype == np.uint8 and image_u.dtype == np.uint8
            drawn_image_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2BGR)
            drawn_image_u = cv2.cvtColor(image_u, cv2.COLOR_RGB2BGR)
            for sub_dir in ['supervised', 'semi_supervised']:
                os.makedirs(osp.join(self._img_save_dir, sub_dir), exist_ok=True)
            cv2.imwrite(osp.join(self._img_save_dir, 'supervised', save_file_name_l), drawn_image_l)
            cv2.imwrite(osp.join(self._img_save_dir, 'semi_supervised', save_file_name_u), drawn_image_u)
            
        else:       
            cv2.imwrite(osp.join(self._img_save_dir, save_file_name_l), drawn_image_l)