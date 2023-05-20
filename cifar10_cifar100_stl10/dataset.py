import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.datasets import CIFAR10, CIFAR100
from random import sample 
import cv2
import numpy as np
import torch
import torchvision.datasets as datasets

class CIFAR10Pair(CIFAR10):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.split=split

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return transforms.ToTensor()(img), pos_1, pos_2, target


class CIFAR100Pair(CIFAR100):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.split=split

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), pos_1, pos_2, target


class STL10Pair(STL10):
    def __init__(self, train, **kwargs):
        super().__init__(**kwargs)
        self.train=train

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return transforms.ToTensor()(img), pos_1, pos_2, target


class CIFAR10_split(CIFAR10):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)

        print("CIFAR10 with unused split arg")

class CIFAR100_split(CIFAR100):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)

        print("CIFAR100 with unused split arg")

class STL10_split(STL10):
    def __init__(self, train, **kwargs):
        super().__init__(**kwargs)

        print("STL10 with unused train arg")

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


# General CIFAR, STL, ImageNet transforms
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]
cifar10_train_transform = transforms.Compose([
                          transforms.RandomResizedCrop(32),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                          transforms.RandomGrayscale(p=0.2),
                          GaussianBlur(kernel_size=int(0.1 * 32)),
                          transforms.ToTensor(),
                          transforms.Normalize(cifar10_mean, cifar10_std)
                          ])

cifar10_test_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(cifar10_mean, cifar10_std)
                         ])

cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]
cifar100_train_transform = transforms.Compose([
                           transforms.RandomResizedCrop(32),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                           transforms.RandomGrayscale(p=0.2),
                           GaussianBlur(kernel_size=int(0.1 * 32)),
                           transforms.ToTensor(),
                           transforms.Normalize(cifar100_mean, cifar100_std)
                           ])

cifar100_test_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(cifar100_mean, cifar100_std)
                          ])

# STL10
stl10_mean = (0.4406, 0.4273, 0.3858)
stl10_std = (0.2312, 0.2265, 0.2237)
stl10_train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        GaussianBlur(kernel_size=int(0.1 * 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(stl10_mean, stl10_std)
                        ])

stl10_test_transform = transforms.Compose([    
                       transforms.Resize(32),
                       transforms.ToTensor(),
                       transforms.Normalize(stl10_mean, stl10_std)
                       ])


__datasets__ = {"cifar10pair": CIFAR10Pair,
                "cifar100pair": CIFAR100Pair,
                "cifar10": CIFAR10_split,
                "cifar100": CIFAR100_split,
                "stl10pair": STL10Pair,
                "stl10": STL10_split,
                }



__datasets_stats__ = {"cifar10": [cifar10_mean, cifar10_std],
                      "cifar100": [cifar100_mean, cifar100_std],
                      "stl10": [stl10_mean, stl10_std]}

def get_dataset(dataset_name, root, args, pair=True):
    if pair:    dataset_name +='pair'
    if dataset_name not in __datasets__.keys(): raise Exception('Invalid dataset name')
    
    if dataset_name == 'cifar10pair' or dataset_name == 'cifar10':
        print("=> Loading CIFAR10 dataset")
        train_data = __datasets__[dataset_name](root=root, train=True, split = 'train', download=True, transform=cifar10_train_transform)
        memory_data = __datasets__[dataset_name](root=root, train=True, split = 'train', download=True, transform=cifar10_test_transform)
        test_data = __datasets__[dataset_name](root=root, train=False, split = 'test', download=True, transform=cifar10_test_transform)

    elif dataset_name == 'cifar100pair' or dataset_name == 'cifar100':
        print("=> Loading CIFAR100 dataset from", root)
        train_data = __datasets__[dataset_name](root=root, train=True, split = 'train', download=True, transform=cifar100_train_transform)
        memory_data = __datasets__[dataset_name](root=root, train=True, split = 'train', download=True, transform=cifar100_test_transform)
        test_data = __datasets__[dataset_name](root=root, train=False, split = 'test', download=True, transform=cifar100_test_transform)

    if dataset_name=='stl10pair' or dataset_name == 'stl10':
        print("=> Loading STL10 dataset")
        train_data = __datasets__[dataset_name](root=root, train=True, split='train+unlabeled' if not args.lin_eval else 'train', download=True, transform=stl10_train_transform)
        memory_data = __datasets__[dataset_name](root=root, train=True, split = 'train', download=True, transform= stl10_test_transform)
        test_data = __datasets__[dataset_name](root=root, train=False, split = 'test', download=True, transform=stl10_test_transform)

    return train_data, memory_data, test_data
        

def TorchGaussianBlur(img):
    kernel_size = int(0.1 * 32)
    prob = np.random.random_sample()
    if prob < 0.5:
        img = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))(img)
    return img

def get_transforms(args):
    t1 = transforms.RandomResizedCrop(32)
    t2 = transforms.RandomHorizontalFlip(p=0.5)
    t3 = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    t4 = transforms.RandomGrayscale(p=0.2)
    t5 = TorchGaussianBlur
    stats = __datasets_stats__[args.dataset_name]
    t6 = transforms.Normalize(stats[0], stats[1])
    return [t1, t2, t3, t4, t5, t6]



