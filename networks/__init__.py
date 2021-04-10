import torch 
import torch.nn as nn

from .encoder import Encoder
from .vq import VQ
from .decoder import Decoder
from .initialize import init_weights


def init_models(input_dim: int = 4,
                output_dim: int = 4,
                emb_dim: int = 64,
                dict_size: int = 512,
                enc_filters: list = [32, 64, 128, 128, 128, 128],
                dec_filters: list = [128, 128, 128, 128, 64, 32],
                latent_size: int = 8,
                init_type: str = 'kaiming',
                faiss_backend: str = 'torch',
                ):

    encoder = Encoder(input_dim=input_dim,
                      emb_dim=emb_dim,
                      filters=enc_filters)

    vq = VQ(emb_dim=emb_dim,
            dict_size=dict_size,
            momentum=0.99,
            eps=1e-5,
            knn_backend=faiss_backend)

    decoder = Decoder(output_dim=output_dim,
                      emb_dim=emb_dim,
                      filters=dec_filters)

    init_weights(encoder, init_type)
    init_weights(decoder, init_type)

    encoder.cuda()
    encoder = nn.DataParallel(encoder)

    vq.cuda()
    vq = nn.DataParallel(vq)

    decoder.cuda()
    decoder = nn.DataParallel(decoder)

    return encoder, vq, decoder


def load_models(encoder_path: str, vq_path: str, decoder_path: str,
                *args, **kwargs):

    encoder, vq, decoder = init_models(*args, **kwargs)

    encoder.load_state_dict(torch.load(encoder_path))
    vq.load_state_dict(torch.load(vq_path))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, vq, decoder
