# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch.nn.functional as F
from torchvision.transforms import functional


class DifferentiableDataAugmentation:
    def __init__(self):
        pass

    def sample_params(self, x, seed=None):
        """ Sample parameters for a given data augmentation """
        return 0

    def apply(self, x, params):
        """ Apply data augmentation to image """
        assert params == 0
        return x

    def __call__(self, x, params):
        return self.apply(x, params)


class All(DifferentiableDataAugmentation):

    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3/4, 4/3), resize_scale=(0.2, 1.0), blur_size=17, flip=True, interpolation='bilinear'):
        """
        Apply a data augmentations, chosen at random between none, rotation, crop, resize, blur, with random parameters.
        
        Args:
            degrees (float): Amplitude of rotation augmentation (in ±degrees)
            crop_scale (tuple of float): Lower and upper bounds for the random area of the crop before resizing
            crop_ratio (tuple of float): Lower and upper bounds for the random aspect ratio of the crop, before resizing
            resize_scale (tuple of float): Lower and upper bounds for the random area of the resizing
            blur_size (int): Upper bound of the size of the blur kernel (sigma=ksize*0.15+0.35 and ksize=(sigma-0.35)/0.15)
            flip (boolean): whether to apply random horizontal flip
        """
        self.degrees = degrees 
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.resize_scale = resize_scale
        self.blur_size = blur_size
        self.flip = flip
        self.interpolation = functional.InterpolationMode(interpolation)

    def sample_params(self, x):
        # randomly select one of augmentations
        ps = np.array([1,1,1,1,1])
        ps = ps / ps.sum()
        augm_type = np.random.choice(['none', 'rotation', 'crop', 'resize', 'blur'], p=ps)
        # flip param
        f = np.random.rand()>0.5 if self.flip else 0  
        # sample params
        if augm_type == 'none':
            return augm_type, 0, f
        elif augm_type == 'rotation':
            d = np.random.vonmises(0, 1)*self.degrees/np.pi
            return augm_type, d, f
        elif augm_type in ['crop', 'resize']:
            width, height = functional.get_image_size(x)
            area = height * width
            target_area = np.random.uniform(*self.crop_scale) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(self.crop_ratio[0]), np.log(self.crop_ratio[1])))
            tw = int(np.round(np.sqrt(target_area * aspect_ratio)))
            th = int(np.round(np.sqrt(target_area / aspect_ratio)))
            if augm_type == 'crop':
                i = np.random.randint(0, max(min(height - th + 1, height-1), 0)+1)
                j = np.random.randint(0, max(min(width - tw + 1, width-1), 0)+1)
                return augm_type, (i ,j, th, tw), f
            elif augm_type == 'resize':
                s = np.random.uniform(*self.resize_scale)
                return augm_type, (s, th, tw), f
        elif augm_type == 'blur':
            b = np.random.randint(1, self.blur_size+1)
            b = b-(1-b%2) # make it odd         
            return augm_type, b, f
        

    def apply(self, x, augmentation):
        augm_type, param, f = augmentation
        if augm_type == 'blur':
            x = functional.gaussian_blur(x, param)
        if augm_type == 'rotation':
            x = functional.rotate(x, param, interpolation=self.interpolation)
            # x = functional.rotate(x, d, expand=True, interpolation=self.interpolation)
        elif augm_type == 'crop':
            x = functional.crop(x, *param)
        elif augm_type == 'resize':
            s, h, w = param
            x = functional.resize(x, int((s**0.5)*min(h,w)), interpolation=self.interpolation)
        x = functional.hflip(x) if f else x
        return x
