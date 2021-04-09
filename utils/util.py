import os
import json
import random
import collections
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def norm(x, vmin=0, vmax=255):
    x -= vmin
    x /= (vmax - vmin)
    x = 2.0 * (x - 0.5)
    x = x.clamp_(-1, 1)
    return x


def denorm(x, vmin=0, vmax=1):
    x = (x + 1) / 2.0
    x = x.clamp_(0, 1)
    x += vmin
    x *= (vmax - vmin)
    return x


def minmax_norm(array, vmin=None, vmax=None):
    if vmin is None:
        vmin = array.min()
    if vmax is None:
        vmax = array.max()
    array -= vmin
    array /= (vmax - vmin)
    return array


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_json(path):
    def _json_object_hook(d):
        for k, v in d.items():
            d[k] = None if v is False else v
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_output_dir_path(config):
    study_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = config.save.study_name + '_' + study_time
    output_dir_path = os.path.join(config.save.output_root_dir, dir_name)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )


def save_config(config, seed, save_dir_path):
    config_to_save = collections.defaultdict(dict)

    for key, child in config._asdict().items():
        for k, v in child._asdict().items():
            config_to_save[key][k] = v

    config_to_save['seed'] = seed
    config_to_save['save_dir_path'] = save_dir_path

    save_path = os.path.join(save_dir_path, 'config.json')
    os.makedirs(save_dir_path, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f)
