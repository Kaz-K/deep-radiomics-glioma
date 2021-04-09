from typing import Optional
from torch.utils import data
from torchvision import transforms

from .transformation import ToTensor
from .transformation import RandomIntensityShiftScale
from .transformation import RandomHorizontalFlip
from .transformation import RandomRotate
from .dataset import MICCAIBraTSDataset


def get_data_loader(dataset_name: str,
                    modalities: list,
                    root_dir_paths: list,
                    augmentation_type: str,
                    use_shuffle: bool,
                    batch_size: int,
                    num_workers: int,
                    initial_randomize: bool = True,
                    patient_ids: Optional[list] = None,
                    ) -> data.DataLoader:

    assert dataset_name == 'MICCAIBraTSDataset'
    assert augmentation_type in {'A', 'B', 'none'}

    if augmentation_type == 'A':
        transform = transforms.Compose([RandomIntensityShiftScale(),
                                        RandomHorizontalFlip(),
                                        ToTensor(),
                                        RandomRotate()])

    elif augmentation_type == 'B':
        transform = transforms.Compose([RandomHorizontalFlip(),
                                        ToTensor()])

    elif augmentation_type == 'none':
        transform = transforms.Compose([ToTensor()])

    dataset = MICCAIBraTSDataset(
        root_dir_paths=root_dir_paths,
        transform=transform,
        modalities=modalities,
        patient_ids=patient_ids,
        initial_randomize=initial_randomize,
    )

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=use_shuffle,
                           num_workers=num_workers,
                           pin_memory=True)
