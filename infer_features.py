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


save_dir_path = './results/infer_features/'


class IntensityShiftScale(object):

    def __init__(self, shift: float, scale: float):
        self.shift = shift
        self.scale = scale

    def __call__(self, image):
        return self.scale * (image + self.shift)


def infer_deep_features(patient_id, transform):
    features = None

    for file in dataset.get_patient_samples(patient_id):
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
    parser.add_argument('-s', '--shift', type=float, default=0.0)
    parser.add_argument('-t', '--scale', type=float, default=0.0)
    args = parser.parse_args()

    shift_range = args.shift
    scale_range = args.scale

    config = load_json(args.config)
    minlength = config.model.dict_size

    encoder, vq, decoder, dataset = init_eval(config)

    encoder.eval()
    vq.eval()
    decoder.eval()

    os.makedirs(save_dir_path, exist_ok=True)

    save_path = os.path.join(
        save_dir_path,
        'features_scale_{:.1f}_shift_{:.1f}.csv'.format(scale_range, shift_range),
    )

    if not os.path.exists(save_path):
        print('Processing: ', save_path)

        transform = IntensityShiftScale(
            shift=random.uniform(-shift_range, shift_range),
            scale=random.uniform(1.0 - scale_range, 1.0 + scale_range),
        )

        features_list = None
        for patient_id in tqdm(dataset.patient_ids):
            features = infer_deep_features(patient_id, transform)
            features_list = concat(features_list, features)

        df = pd.DataFrame(data=features_list, index=dataset.patient_ids)
        df.to_csv(save_path)
