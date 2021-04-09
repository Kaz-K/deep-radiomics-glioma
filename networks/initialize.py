# Originally written by ozan-oktay
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks_other.py

import torch.nn as nn
from torch.nn import init


__all__ = ['init_weights']


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.BatchNorm1d:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.Linear:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Module:
        weights_init_normal(m)


def weights_init_kaiming(m):
    if type(m) == nn.Conv2d:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.BatchNorm2d:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.Module:
        weights_init_kaiming(m)


def init_weights(net, init_type='kaiming'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
