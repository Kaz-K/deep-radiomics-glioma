import torch
import argparse
import random
import radiomics
import numpy as np
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


def eval_reproducibility(patient_id, transform):
    src_features = None
    dst_features = None

    for file in tqdm(dataset.get_patient_samples(patient_id)):
        sample = dataset.load_file(file)
        image = sample['image'].unsqueeze(0)
        transformed = transform(image)

        with torch.no_grad():
            _, _, s_ids = vq(encoder(image))
            _, _, d_ids = vq(encoder(transformed))

        s_ids = s_ids.cpu().numpy().ravel()
        d_ids = d_ids.cpu().numpy().ravel()

        src_features = concat(src_features, s_ids)
        dst_features = concat(dst_features, d_ids)

    src_features = np.bincount(src_features.ravel(), minlength=minlength)
    dst_features = np.bincount(dst_features.ravel(), minlength=minlength)

    diff = np.abs(src_features - dst_features).sum()
    denominator = src_features.sum()

    return diff / denominator


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate reproducibility')
    parser.add_argument('-c', '--config', help='evaluation config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    minlength = config.model.dict_size

    encoder, vq, decoder, dataset = init_eval(config)

    encoder.eval()
    vq.eval()
    decoder.eval()

    for shift_range in np.arange(0.0, 1.1, 0.1):
        scale_range = 0.0

        transform = IntensityShiftScale(
            shift=random.uniform(-shift_range, shift_range),
            scale=random.uniform(1.0 - scale_range, 1.0 + scale_range),
        )

        values = []
        for patient_id in dataset.patient_ids:
            ratio = eval_reproducibility(patient_id, transform)
            values.append(ratio)

        values_mean = np.mean(values)
        values_std = np.std(values)

        print(shift_range, values_mean, values_std)
