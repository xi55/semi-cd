# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample



@TRANSFORMS.register_module()
class PackSemiInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('imgs_l', 'imgs_u', 'gt_seg_map', 'ori_shape','img_shape', 'img_path_l', 'img_path_u',
                            'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'scale', 'keep_ratio'
                            )):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'imgs_l' in results:
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img
            
            imgs_l = [_transform_img(img_l) for img_l in results['imgs_l']]
            imgs_l = torch.cat(imgs_l, axis=0) # -> (6, H, W)
            packed_results['inputs_l'] = imgs_l

        if 'imgs_u' in results:
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img
            
            imgs_u = [_transform_img(img_u) for img_u in results['imgs_u']]
            imgs_u = torch.cat(imgs_u, axis=0) # -> (6, H, W)
            packed_results['inputs_u'] = imgs_u
        
        if 'imgs_u_s' in results:
            if results['imgs_u_s'] is not None:
                def _transform_img(img):
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    if not img.flags.c_contiguous:
                        img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                    else:
                        img = img.transpose(2, 0, 1)
                        img = to_tensor(img).contiguous()
                    return img
                
                imgs_u_s = [_transform_img(img_u_s) for img_u_s in results['imgs_u_s']]
                imgs_u_s = torch.cat(imgs_u_s, axis=0) # -> (6, H, W)
                packed_results['inputs_u_s'] = imgs_u_s
            else:
                packed_results['inputs_u_s'] = None

        if 's2_img' in results:
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img
            
            imgs = [_transform_img(img) for img in results['s2_img']]
            imgs = torch.cat(imgs, axis=0) # -> (6, H, W)
            packed_results['inputs_s2'] = imgs

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        if 'label_u' in results:
            label_u_data = dict(
                data=to_tensor(results['label_u'][None,
                                                     ...].astype(np.int64)))
            data_sample.label_u = PixelData(**label_u_data)
        
        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))
        
        if 'gt_seg_map_from' in results:
            gt_sem_seg_data_from = dict(
                data=to_tensor(results['gt_seg_map_from'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_from=PixelData(**gt_sem_seg_data_from)))

        if 'gt_seg_map_to' in results:
            gt_sem_seg_data_to = dict(
                data=to_tensor(results['gt_seg_map_to'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_to=PixelData(**gt_sem_seg_data_to)))

        if 'label_seg_map' in results:
            gt_sem_seg = dict(
                data=to_tensor(results['label_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(label_seg_map=PixelData(**gt_sem_seg)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

