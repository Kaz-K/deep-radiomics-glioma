import os
import numpy as np
import collections
from typing import Tuple

import torch
import pytorch_lightning as pl

from dataio import get_data_loader
from networks import init_models
from functions import SoftDiceLoss
from functions import FocalLoss
from functions import DiceCoefficient
from functions import OneHotEncoder
from utils import minmax_norm


class TumorSegmentation(pl.LightningModule):

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

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                               chain(self.encoder.parameters(),
                                     self.vq.parameters(),
                                     self.decoder.parameters())),
                             self.config.optimizer.lr, [0.9, 0.9999],
                             weight_decay=self.config.optimizer.weight_decay)
        return [optimizer], []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = self.encoder(x)
        qlat, l_lat, ids = self.vq(lat)
        logit = self.decoder(qlat)
        return logit, l_lat

    def l_seg(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = self.one_hot_encoder(label)
        dice_loss = self.l_dice(logit, target)
        focal_loss = self.l_focal(logit, target)
        return dice_loss + focal_loss

    def training_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']

        logit, l_lat = self.forward(image)

        lat_loss = self.config.loss_weight.w_lat * l_lat
        seg_loss = self.config.loss_weight.w_seg * self.l_seg(logit, label)

        total_loss = (lat_loss + seg_loss).sum()

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total_loss', total_loss.sum(), prog_bar=True)
        self.log('lat_loss', lat_loss.sum(), prog_bar=True)
        self.log('seg_loss', seg_loss.sum(), prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']

        with torch.no_grad():
            logit, _ = self.forward(image)

        dice = self.dice_metric(logit, label)

        if batch_idx == 0:
            image = image.detach().cpu()
            image = minmax_norm(image)

            label = label.detach().cpu()
            output = logit.argmax(dim=1).detach().cpu()

            n_images = min(self.config.save.n_save_images, image.size(0))

            save_modalities = ['t1ce']
            if 'flair' in self.config.dataset.modalities:
                save_modalities.append('flair')

            save_series = []
            for modality in save_modalities:
                idx = self.config.dataset.modalities.index(modality)
                save_image = image[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                save_series.append(save_image)

            label = label[:n_images, ...].float()[:, np.newaxis, ...]
            output = output[:n_images, ...].float()[:, np.newaxis, ...]

            max_label_val = self.config.metric.n_classes - 1
            label /= max_label_val
            output /= max_label_val

            save_series.append(label)
            save_series.append(output)

            label_grid = torch.cat(save_series)
            self.logger.log_images('segmentation',
                                   label_grid,
                                   self.current_epoch,
                                   self.global_step,
                                   nrow=n_images)

        return dice
