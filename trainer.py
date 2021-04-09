import os
import numpy as np
import collections

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from dataio import get_data_loader
from networks import init_models
from functions import SoftDiceLoss
from functions import FocalLoss
from functions import DiceCoefficient
from functions import OneHotEncoder
from utils import minmax_norm


class SegmentationTrainer(pl.LightningModule):

    def __init__(self, config: collections.namedtuple, save_dir_path: str) -> None:
        super().__init__()

        self.config = config
        self.save_dir_path = save_dir_path

        self.class_name_to_index = self.config.metric.class_name_to_index._asdict()
        self.index_to_class_name = {v: k for k, v in self.class_name_to_index.items()}

        self.encoder, self.vq, self.decoder = init_models(
            input_dim=self.config.model.input_dim,
            output_dim=self.config.model.output_dim,
            emb_dim=self.config.model.emb_dim,
            dict_size=self.config.model.dict_size,
            enc_filters=self.config.model.enc_filters,
            dec_filters=self.config.model.dec_filters,
            latent_size=self.config.model.latent_size,
            init_type=self.config.model.init_type,
            faiss_backend=self.config.model.faiss_backend,
        )

        self.l_dice = SoftDiceLoss(ignore_index=self.config.dice_loss.ignore_index)
        self.l_focal = FocalLoss(gamma=self.config.focal_loss.gamma,
                                 alpha=self.config.focal_loss.alpha)

        self.one_hot_encoder = OneHotEncoder(n_classes=self.config.metric.n_classes).forward

        self.dice_metric = DiceCoefficient(n_classes=self.config.metric.n_classes,
                                           index_to_class_name=self.index_to_class_name)

    def export_models(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.vq.state_dict(), os.path.join(path, 'vq.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'decoder.pth'))

    def train_dataloader(self):
        data_loader = get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type=self.config.dataset.augmentation_type,
            use_shuffle=self.config.dataset.use_shuffle,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
        )
        return data_loader

    def val_dataloader(self):
        return get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type='none',
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
        )

    def test_dataloader(self):
        return get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type='none',
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            initial_randomize=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = self.encoder(x)
        qlat, lat_loss, ids = self.vq(lat)
        logit = self.decoder(qlat)
        return logit, lat_loss

    def l_seg(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = self.one_hot_encoder(label)
        dice_loss = self.l_dice(logit, target)
        focal_loss = self.l_focal(logit, target)
        return dice_loss + focal_loss


    def training_step(self, batch: dict, batch_idx: int):
        image = batch['image']
        label = batch['label']

        logit, lat_loss = self.forward(image)

        latent_loss = self.config.loss_weight.w_latent * lat_loss
        seg_loss = self.config.loss_weight.w_seg * self.l_seg(logit, label)

        total_loss = (latent_loss + seg_loss).sum()

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total_loss', total_loss.sum(), prog_bar=True)
        self.log('latent_loss', latent_loss.sum(), prog_bar=True)
        self.log('seg_loss', seg_loss.sum(), prog_bar=True)

        return total_loss
