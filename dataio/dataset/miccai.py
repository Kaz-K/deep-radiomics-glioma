import os
import numpy as np
import random
from torch.utils import data
from torchvision import transforms
from typing import Optional


class MICCAIBraTSDataset(data.Dataset):
    n_slices = 155

    def __init__(self,
                 root_dir_paths: list,
                 transform: transforms.Compose,
                 modalities: list,
                 patient_ids: Optional[list] = None,
                 initial_randomize: bool = True,
                 ) -> None:
        super().__init__()

        self.patient_ids = patient_ids
        self.transform = transform
        self.modalities = modalities
        self.files = self.build_files(root_dir_paths, initial_randomize)

    def build_files(self, root_dir_paths: list, initial_randomize: bool) -> list:
        files = []

        for root_dir_path in root_dir_paths:
            for patient_id in os.listdir(root_dir_path):
                patient_dir_path = os.path.join(root_dir_path, patient_id)

                if self.patient_ids is not None:
                    if patient_id not in self.patient_ids:
                        continue

                for n_slice in range(self.n_slices):
                    series = {
                        'patient_id': patient_id,
                        'n_slice': n_slice,
                    }

                    for modality in self.modalities + ['seg']:
                        file_path = os.path.join(
                            patient_dir_path,
                            patient_id + '_' + modality + '_' + str(n_slice).zfill(4) + '.npy'
                        )
                        assert os.path.exists(file_path)
                        series[modality] = file_path

                    files.append(series)

        if initial_randomize:
            random.shuffle(files)

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        series = []
        for modality in self.modalities:
            series.append(
                np.load(file[modality]).astype(np.float32)[np.newaxis, ...]
            )
        image = np.concatenate(series, axis=0)

        seg = np.load(file['seg']).astype(np.int32)
        seg[seg == 4] = 3

        sample = {
            'patient_id': file['patient_id'],
            'n_slice': file['n_slice'],
            'image': image,
            'label': seg,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
