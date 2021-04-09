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

        if 'seg_label' in sample.keys():
            label = sample['seg_label'].float().unsqueeze(0).unsqueeze(0)

        B, _, H, W = image.shape
        rotated_image, transform = self.rotate(image)

        if 'seg_label' in sample.keys():
            rotated_label = K.warp_perspective(label, transform, dsize=(H, W), flags='nearest')
            rotated_label = rotated_label.int()

            sample.update({
                'image': rotated_image[0, ...],
                'seg_label': rotated_label[0, 0, ...],
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

        if 'seg_label' in sample.keys():
            seg_label = sample['seg_label']

        if random.uniform(0, 1) < 0.5:
            image = copy.deepcopy(np.flip(image, axis=2))

            if 'seg_label' in sample.keys():
                seg_label = copy.deepcopy(np.flip(seg_label, axis=1))

        if 'seg_label' in sample.keys():
            sample.update({
                'image': image,
                'seg_label': seg_label,
            })

        else:
            sample.update({
                'image': image,
            })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict) -> dict:
        image = sample['image']

        if 'seg_label' in sample.keys():
            seg_label = sample['seg_label']

        if 'class_label' in sample.keys():
            class_label = sample['class_label']

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        image = torch.from_numpy(image).float()

        if 'seg_label' in sample.keys():
            seg_label = torch.from_numpy(seg_label).int()

        if 'class_label' in sample.keys():
            assert isinstance(class_label, int)
            class_label = np.asarray([class_label])
            class_label = torch.from_numpy(class_label).long()

        sample.update({
            'image': image,
        })

        if 'seg_label' in sample.keys():
            sample.update({
                'seg_label': seg_label,
            })

        if 'class_label' in sample.keys():
            sample.update({
                'class_label': class_label,
            })

        return sample
