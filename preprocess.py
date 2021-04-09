import os
import numpy as np
import nibabel as nib
import scipy.misc
import cv2


IMAGE_SIZE = 256

cfg_miccai_brats_training_hgg = {
    'src_dir_path': './data/MICCAI_BraTS_2019_Data_Training/HGG',
    'dst_dir_path': './data/MICCAI_BraTS_2019_Data_Training_Slices/HGG',
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}

cfg_miccai_brats_training_lgg = {
    'src_dir_path': './data/MICCAI_BraTS_2019_Data_Training/HGG',
    'dst_dir_path': './data/MICCAI_BraTS_2019_Data_Training_Slices/HGG',
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}

cfg_miccai_brats_testing = {
    'src_dir_path': './data/MICCAI_BraTS_2019_Data_Testing',
    'dst_dir_path': './data/MICCAI_BraTS_2019_Data_Testing_Slices',
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}

cfg_miccai_brats_validation = {
    'src_dir_path': './data/MICCAI_BraTS_2019_Data_Validation',
    'dst_dir_path': './data/MICCAI_BraTS_2019_Data_Validation_Slices',
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}


def z_score_normalize(array):
    array = array.astype(np.float32)
    mask = array > 0
    mean = np.mean(array[mask])
    std = np.std(array[mask])
    array -= mean
    array /= std
    return array


def preprocess(config, patient_id):

    patient_dir_path = os.path.join(
        config['src_dir_path'], patient_id,
    )

    for modality in config['modalities']:
        file_path = os.path.join(
            patient_dir_path,
            patient_id + '_' + modality['pattern'] + '.nii.gz'
        )
        nii_file = nib.load(file_path)
        series = nii_file.get_data()

        if modality['name'] == 'SEG':
            series = series.astype(np.int32)
            bincount = np.bincount(series.ravel())

            if 'Training' in config['src_dir_path']:
                if len(bincount) > 3:
                    assert bincount[3] == 0

                series[series == 4] = 3  # 3: ET (GD-enhancing tumor)
                series[series == 2] = 2  # 2: ED (peritumoral edema)
                series[series == 1] = 1  # 1: NCR/NET (non-enhancing tumor core)
                series[series == 0] = 0  # 0: Background

        else:
            # Perform volume-wise Z-score normalization.
            series = z_score_normalize(series)

        for i in range(series.shape[2]):
            slice = series[..., i]
            slice = np.rot90(slice, k=3)

            # Perform image resizing.
            if modality['name'] == 'SEG':
                slice = scipy.misc.imresize(
                    slice.astype(np.uint8),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interp='nearest',
                    mode='L',
                )
            else:
                slice = cv2.resize(slice, (IMAGE_SIZE, IMAGE_SIZE))

            dst_patient_dir_path = os.path.join(
                config['dst_dir_path'], patient_id
            )
            os.makedirs(dst_patient_dir_path, exist_ok=True)

            save_path = os.path.join(
                dst_patient_dir_path,
                patient_id + '_' + modality['save_pattern'] + '_' + str(i).zfill(4) + '.npy'
            )

            np.save(save_path, slice)


if __name__ == '__main__':

    configs = [
        cfg_miccai_brats_training_hgg,
        cfg_miccai_brats_training_lgg,
        cfg_miccai_brats_testing,
        cfg_miccai_brats_validation,
    ]

    for config in configs:
        patient_ids = os.listdir(config['src_dir_path'])

        for patient_id in patient_ids:
            preprocess(config, patient_id)
