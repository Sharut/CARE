# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
from torchvision import transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, std_transform = None):
        self.base_transform = base_transform
        self.std_transform = std_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        if self.std_transform is not None:
            r = self.std_transform(x)
        else:   r = None
        return [q, k, r]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TorchGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, kernel_size, sigma=[.1, 2.]):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        x = transforms.GaussianBlur(self.kernel_size, sigma=(self.sigma[0], self.sigma[1]))(x)
        return x

