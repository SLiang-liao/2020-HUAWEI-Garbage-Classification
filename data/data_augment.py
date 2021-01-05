import math

import cv2
import numpy as np
import random
import torch

def _crop(image):
    mode =  random.choice(
        (
            None,
            # (xmin, ymin, xmax, ymax)
            (0, 0, 0.8, 0.8),    # top-left
            (0.2, 0, 1, 0.8),    # top-right
            (0, 0.2, 0.8, 1),    # bottom-letf
            (0.2, 0.2, 1, 1),    # bottom-right    
            (0.1, 0.1, 0.9, 0.9),# center
        )
    )
    if mode == None:
        return image

    height, width, _ = image.shape
    img = image[int(height * mode[1]): int(height * mode[3]),
                 int(width * mode[0]): int(width * mode[2]), :]
    return img

def _distort(image, p):
    if random.random() > p:
        return image
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def _expand(image, fill, p):
    if random.random() > p:
        return image

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image


def _mirror(image):
    if random.randrange(2):
        image = image[:, ::-1]
    return image


def preproc_for_test(image, insize, mean, std=(1, 1, 1)):

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    image /= std

    return image

class PreprocessTransform(object):

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), p=0.3):
        self.resize = resize
        self.rgb_means = rgb_means
        self.rgb_std = rgb_std
        self.p = p

    def __call__(self, image):
        img_o = image.copy()
        img = _crop(img_o)
        img = _distort(img, 1)
        img = _expand(img, self.rgb_means, self.p)
        img = _mirror(img)
        img = preproc_for_test(img, self.resize, self.rgb_means, self.rgb_std)
        return img

class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        rgb_std: std of the dataset
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1)):
        self.means = rgb_means
        self.resize = resize
        self.std = rgb_std

    # assume input is cv2 img for now
    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img /= self.std
        return img


