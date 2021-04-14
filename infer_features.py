import os
import torch
import argparse
import random
import radiomics
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_json
from eval_segmentation import init_eval
from eval_segmentation import concat


class IntensityShiftScale(object):

    def __init__(self, shift: float, scale: float):
        self.shift = shift
        self.scale = scale

    def __call__(self, image):
        return self.scale * (image + self.shift)


def infer_deep_features(patient_id, transform):
    features = None

    for file in tqdm(dataset.get_patient_samples(patient_id)):
        sample = dataset.load_file(file)
        image = sample['image'].unsqueeze(0)
        image = transform(image)

        with torch.no_grad():
            _, _, ids = vq(encoder(image))

        ids = ids.cpu().numpy().ravel()
        features = concat(features, ids)

    return np.bincount(features.ravel(), minlength=minlength)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Infer deep features')
    parser.add_argument('-c', '--config', help='evaluation config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    minlength = config.model.dict_size

    encoder, vq, decoder, dataset = init_eval(config)

    encoder.eval()
    vq.eval()
    decoder.eval()

    for shift_range in np.arange(0.0, 1.1, 0.1):
        for scale_range in np.arange(0.0, 1.1, 0.1):

            transform = IntensityShiftScale(
                shift=random.uniform(-shift_range, shift_range),
                scale=random.uniform(1.0 - scale_range, 1.0 + scale_range),
            )

            save_dir_path = './results/infer_features/'
            os.makedirs(save_dir_path, exist_ok=True)

            save_path = os.path.join(
                save_dir_path,
                'features_scale_{:.1f}_shift_{:.1f}.csv'.format(scale_range, shift_range),
            )

            if os.path.exists(save_path):
                continue

            features_list = None
            for patient_id in dataset.patient_ids:
                features = infer_deep_features(patient_id, transform)
                features_list = concat(features_list, features)

            df = pd.DataFrame(data=features_list, index=dataset.patient_ids)
            df.to_csv(save_path)
