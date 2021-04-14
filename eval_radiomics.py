import os
import radiomics
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from operator import itemgetter


data_root_paths = [
    '../Dataset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG',
    '../Dataset/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/LGG',
]


def get_patient_dir_paths(data_root_paths):
    files = []
    for data_root_path in data_root_paths:
        for patient_id in os.listdir(data_root_path):
            patient_dir_path = os.path.join(data_root_path, patient_id)
            files.append({
                'patient_id': patient_id,
                'patient_dir_path': patient_dir_path,
            })
    return files


def numpy_to_simpleitk(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    assert array.ndim == 3
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def calc_radiomics(image, mask):
    image = numpy_to_simpleitk(image)
    mask = numpy_to_simpleitk(mask)

    second_order = radiomics.shape.RadiomicsShape(image, mask)
    second_order.enableAllFeatures()
    second_order.execute()

    measure = {}
    for key, value in second_order.featureValues.items():
        measure.update({
            key: value,
        })

    return measure


if __name__ == '__main__':

    patient_files = get_patient_dir_paths(data_root_paths)
    patient_files = sorted(patient_files, key=itemgetter('patient_id'))

    result = None
    for patient_file in tqdm(patient_files):
        patient_id = patient_file['patient_id']
        patient_dir_path = patient_file['patient_dir_path']

        image_path = os.path.join(patient_dir_path, patient_id + '_t1ce.nii.gz')
        image = nib.load(image_path).get_fdata()

        label_path = os.path.join(patient_dir_path, patient_id + '_seg.nii.gz')
        label = nib.load(label_path).get_fdata()
        label[label > 0] = 1

        second_order = calc_radiomics(image, label)
        second_order.update({
            'patient_id': patient_id,
        })

        second_order = pd.Series(second_order)

        if result is None:
            result = second_order
        else:
            result = pd.concat([result, second_order], axis=1)

    result = result.transpose()
    result = result.set_index('patient_id')

    save_dir_path = './results/radiomics/'
    os.makedirs(save_dir_path, exist_ok=True)

    save_path = os.path.join(save_dir_path, 'radiomics.csv')
    result.to_csv(save_path)
