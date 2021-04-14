import torch
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_json
from dataio import MICCAIBraTSDataset
from dataio import ToTensor
from networks import load_models
from utils import minmax_norm


encoder_path = './saved_models/encoder.pth'
vq_path = './saved_models/vq.pth'
decoder_path = './saved_models/decoder.pth'


def init_eval(config):
    encoder, vq, decoder = load_models(encoder_path, vq_path, decoder_path,
                                       input_dim=config.model.input_dim,
                                       output_dim=config.model.output_dim,
                                       emb_dim=config.model.emb_dim,
                                       dict_size=config.model.dict_size,
                                       enc_filters=config.model.enc_filters,
                                       dec_filters=config.model.dec_filters,
                                       latent_size=config.model.latent_size,
                                       init_type=config.model.init_type,
                                       faiss_backend=config.model.faiss_backend)

    dataset = MICCAIBraTSDataset(root_dir_paths=config.dataset.root_dir_paths,
                                 transform=transforms.Compose([ToTensor()]),
                                 modalities=config.dataset.modalities,
                                 initial_randomize=False)

    return encoder, vq, decoder, dataset


def concat(arr1, arr2):
    arr2 = arr2[np.newaxis, ...]
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.concatenate((arr1, arr2), axis=0)
    return arr1


def calc_dice(label, output, id_to_label):
    dice = {}
    for key in id_to_label.keys():
        ls = label == key
        os = output == key

        if np.any(ls):
            inter = np.sum(ls * os)
            union = np.sum(ls) + np.sum(os)
            score = 2.0 * inter / (union + 1e-5)
        else:
            score = None

        dice.update({
            id_to_label[key]: score,
        })

    return dice


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('-c', '--config', help='evaluation config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)

    class_name_to_index = config.metric.class_name_to_index._asdict()
    index_to_class_name = {v: k for k, v in class_name_to_index.items()}

    encoder, vq, decoder, dataset = init_eval(config)

    result_summary = {
        'patient_id': [],
        'Background': [],
        'NET': [],
        'ED': [],
        'ET': [],
    }

    for patient_id in dataset.patient_ids:
        label_volume = None
        output_volume = None

        for file in tqdm(dataset.get_patient_samples(patient_id)):
            sample = dataset.load_file(file)
            patient_id = sample['patient_id']
            n_slice = sample['n_slice']
            image = sample['image'].unsqueeze(0)
            label = sample['label'].unsqueeze(0)

            with torch.no_grad():
                lat = encoder(image)
                qlat, l_lat, ids = vq(lat)
                logit = decoder(qlat)

            image = image.detach().cpu()[0, 1, ...]
            image = minmax_norm(image)
            label = label.detach().cpu()[0, ...]
            output = logit.argmax(dim=1).detach().cpu()[0, ...]

            label_volume = concat(label_volume, label)
            output_volume = concat(output_volume, output)

        dice_result = calc_dice(label_volume, output_volume, index_to_class_name)
        dice_result.update({
            'patient_id': patient_id,
        })

        result_summary['patient_id'].append(dice_result['patient_id'])
        result_summary['Background'].append(dice_result['Background'])
        result_summary['NET'].append(dice_result['NET'])
        result_summary['ED'].append(dice_result['ED'])
        result_summary['ET'].append(dice_result['ET'])

    df = pd.DataFrame.from_dict(result_summary)
    df.to_csv('segmentation_result.csv')
