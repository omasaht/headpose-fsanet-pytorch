"""
Augmentation Transforms for FSANet Training
Implemented by Omar Hassan to match Original Author's Implementation
https://github.com/shamangary/FSA-Net/blob/master/training_and_testing/TYY_generators.py
August, 2020
"""

import torch
import cv2
import numpy as np
from zoom_transform import _apply_random_zoom

class Normalize(object):
    """Applies following normalization: out =  (img-mean)/std ."""
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = (img-self.mean)/self.std
        return img

class RandomCrop(object):
    """Select random crop portion from input image."""
    def __init__(self):
        pass

    def __call__(self, img):
        dn = np.random.randint(15,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]
        h = img.shape[0]
        w = img.shape[1]
        out = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)

        return out

class RandomCropBlack(object):
    """
    Select random crop portion from input image.
    Paste crop region on a black image having same shape as input image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        dn = np.random.randint(15,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]

        h = img.shape[0]
        w = img.shape[1]

        dx_shift = np.random.randint(dn,size=1)[0]
        dy_shift = np.random.randint(dn,size=1)[0]
        out = np.zeros_like(img)
        out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        return out

class RandomCropWhite(object):
    """
    Select random crop portion from input image.
    Paste crop region on a white image having same shape as input image.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        dn = np.random.randint(15,size=1)[0]+1

        dx = np.random.randint(dn,size=1)[0]
        dy = np.random.randint(dn,size=1)[0]

        h = img.shape[0]
        w = img.shape[1]

        dx_shift = np.random.randint(dn,size=1)[0]
        dy_shift = np.random.randint(dn,size=1)[0]
        out = np.ones_like(img)*255
        out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = img[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]

        return out

class RandomZoom(object):
    """Apply RandomZoom transformation."""
    def __init__(self,zoom_range=[0.8,1.2]):
        self.zoom_range = zoom_range

    def __call__(self,img):
        out = _apply_random_zoom(img,self.zoom_range)

        return out

class SequenceRandomTransform(object):
    """
    Apply Transformation in a sequenced random order
    similar to original author's implementation
    """
    def __init__(self,zoom_range=[0.8,1.2]):
        self.rc = RandomCrop()
        self.rcb = RandomCropBlack()
        self.rcw = RandomCropWhite()
        self.rz = RandomZoom(zoom_range=zoom_range)

    def __call__(self,img):
        rand_r = np.random.random()
        if  rand_r < 0.25:
            img = self.rc(img)

        elif rand_r >= 0.25 and rand_r < 0.5:
            img = self.rcb(img)

        elif rand_r >= 0.5 and rand_r < 0.75:
            img = self.rcw(img)

        if np.random.random() > 0.3:
            img = self.rz(img)

        return img

class ToTensor(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)
