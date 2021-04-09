import torch
import random
import copy
import numpy as np
import kornia as K
from kornia.augmentation import RandomRotation


class RandomRotate(object):

    def __init__(self, degree: float = 10.0) -> None:
        self.rotate = RandomRotation(degrees=degree, return_transform=True)

    def __call__(self, sample: dict) -> dict:
        image = sample['image'].unsqueeze(0)

        if 'label' in sample.keys():
            label = sample['label'].float().unsqueeze(0).unsqueeze(0)

        B, _, H, W = image.shape
        rotated_image, transform = self.rotate(image)

        if 'label' in sample.keys():
            rotated_label = K.warp_perspective(label, transform, dsize=(H, W), flags='nearest')
            rotated_label = rotated_label.int()

            sample.update({
                'image': rotated_image[0, ...],
                'label': rotated_label[0, 0, ...],
            })

        else:
            sample.update({
                'image': rotated_image[0, ...],
            })

        return sample


class RandomIntensityShiftScale(object):

    def __call__(self, sample: dict) -> dict:
        image = sample['image']

        for i in range(image.shape[0]):
            shift = random.uniform(-0.1, 0.1)
            scale = random.uniform(0.9, 1.1)
            img = image[i, ...]
            image[i, ...] = scale * (img + shift)

        sample.update({
            'image': image,
        })

        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample: dict) -> dict:
        image = sample['image']

        if 'label' in sample.keys():
            label = sample['label']

        if random.uniform(0, 1) < 0.5:
            image = copy.deepcopy(np.flip(image, axis=2))

            if 'label' in sample.keys():
                label = copy.deepcopy(np.flip(label, axis=1))

        if 'label' in sample.keys():
            sample.update({
                'image': image,
                'label': label,
            })

        else:
            sample.update({
                'image': image,
            })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict) -> dict:
        image = sample['image']

        if 'label' in sample.keys():
            label = sample['label']

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        image = torch.from_numpy(image).float()

        if 'label' in sample.keys():
            label = torch.from_numpy(label).int()

        sample.update({
            'image': image,
        })

        if 'label' in sample.keys():
            sample.update({
                'label': label,
            })

        return sample
