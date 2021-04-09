import os
import fsspec
import json
import collections
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Dict
from typing import Tuple

import torch
from torchvision.utils import save_image
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint


pathlike = Union[Path, str]


def get_filesystem(path: pathlike):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


class ModelSaver(ModelCheckpoint):

    def __init__(self, limit_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_num = limit_num

    def save_checkpoint(self, trainer, pl_module):
        """
        Performs the main logic around saving a checkpoint.
        This method runs on all ranks, it is the responsibility of `self.save_function`
        to handle correct behaviour in distributed training, i.e., saving only on rank 0.
        """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if (
            self.save_top_k == 0  # no models are saved
            or self.period < 1  # no models are saved
            or (epoch + 1) % self.period  # skip epoch
            or trainer.running_sanity_check  # don't save anything during sanity check
            or self.last_global_step_saved == global_step  # already saved at the last step
        ):
            return

        self._add_backward_monitor_support(trainer)
        self._validate_monitor_key(trainer)

        # track epoch when ckpt was last checked
        self.last_global_step_saved = global_step

        # what can be monitored
        monitor_candidates = self._monitor_candidates(trainer)

        # ie: path/val_loss=0.5.ckpt
        filepath = self._get_metric_interpolated_filepath_name(epoch, monitor_candidates)

        # callback supports multiple simultaneous modes
        # here we call each mode sequentially
        # Mode 1: save all checkpoints OR only the top k
        if self.save_top_k:
            self._save_top_k_checkpoints(monitor_candidates, trainer, pl_module, epoch, filepath)

        # Mode 2: save the last checkpoint
        self._save_last_checkpoint(trainer, pl_module, epoch, monitor_candidates, filepath)

        self._delete_old_checkpoint()

    def _delete_old_checkpoint(self):
        checkpoints = sorted([c for c in os.listdir(self.dirpath) if 'ckpt-epoch' in c])
        if len(checkpoints) > self.limit_num:
            margin = len(checkpoints) - self.limit_num
            checkpoints_for_delete = checkpoints[:margin]

            for ckpt in checkpoints_for_delete:
                ckpt_epoch = int(ckpt[len("ckpt-epoch="): len("ckpt-epoch=") + 4])
                if (ckpt_epoch + 1) % 10 != 0:
                    model_path = os.path.join(self.dirpath, ckpt)
                    self._del_model(model_path)


class Logger(LightningLoggerBase):

    def __init__(self,
                 save_dir: str,
                 config: collections.defaultdict,
                 seed: int,
                 monitoring_metrics: list,
                 name: Optional[str]='default',
                 version: Optional[Union[int, str]] = None,
                 **kwargs) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._config = config
        self._seed = seed
        self._version = version
        self._fs = get_filesystem(save_dir)
        self._experiment = None
        self._monitoring_metrics = monitoring_metrics
        self._kwargs = kwargs

    @property
    def root_dir(self) -> str:
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        else:
            return os.path.join(self.save_dir, self.name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self.save_dir, self.name)

        if not self._fs.isdir(root_dir):
            print('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for listing in self._fs.listdir(root_dir):
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace('/', '')
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        values = []
        for key in self._monitoring_metrics:
            if key in metrics.keys():
                v = metrics[key]
                if isinstance(v, torch.Tensor):
                    v = str(v.sum().item())
                else:
                    v = str(v)
            else:
                v = ''
            values.append(v)

        fname = os.path.join(self.log_dir, 'log.csv')
        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(self._monitoring_metrics), file=f)
            print(','.join(values), file=f)

    @rank_zero_only
    def log_val_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'val_logs.csv')
        columns = metrics.keys()
        values = [str(value) for value in metrics.values()]

        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(columns), file=f)
            print(','.join(values), file=f)

    @rank_zero_only
    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        fname = os.path.join(self.log_dir, 'test_logs.csv')
        columns = metrics.keys()
        values = [str(value) for value in metrics.values()]

        with open(fname, 'a') as f:
            if f.tell() == 0:
                print(','.join(columns), file=f)
            print(','.join(values), file=f)

        print('Test results are saved: {}'.format(fname))

    @rank_zero_only
    def log_hyperparams(self, params):
        config_to_save = collections.defaultdict(dict)

        for key, child in self._config._asdict().items():
            for k, v in child._asdict().items():
                config_to_save[key][k] = v

        config_to_save['seed'] = self._seed
        config_to_save['save_dir_path'] = self.log_dir

        save_path = os.path.join(self.log_dir, 'config.json')
        os.makedirs(self.log_dir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_to_save,
                      f,
                      ensure_ascii=False,
                      indent=2,
                      sort_keys=False,
                      separators=(',', ': '))

    @rank_zero_only
    def log_images(self, image_name: str, image: torch.Tensor, current_epoch: int, global_step: int, nrow: int) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        save_path = os.path.join(self.log_dir, f'{image_name}_{current_epoch:04d}_{global_step:06d}.png')
        save_image(image.data, save_path, nrow=nrow)

    def experiment(self):
        return self

    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()
