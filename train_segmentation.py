import os
import random
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from trainers import TumorSegmentation
from utils import load_json
from utils import Logger
from utils import ModelSaver


def build_trainer(config, seed, args):

    monitoring_metrics = ['epoch', 'iteration', 'total_loss', 'latent_loss',
                          'seg_loss', 'NET', 'ED', 'ET']

    logger = Logger(save_dir=config.save.save_dir,
                    config=config,
                    seed=seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics)

    save_dir_path = logger.log_dir

    checkpoint_callback = ModelSaver(limit_num=10,
                                     monitor=None,
                                     filepath=os.path.join(save_dir_path, 'ckpt-{epoch:04d}-{total_loss:.2f}'),
                                     save_top_k=-1)

    if config.run.resume_checkpoint:
        print('Training will resume from: {}'.format(config.run.resume_checkpoint))
        model = TumorSegmentation.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
        )
    else:
        model = TumorSegmentation(config, save_dir_path)

    trainer = pl.Trainer(gpus=config.run.visible_devices,
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=1,
                         automatic_optimization=True,
                         distributed_backend=config.run.distributed_backend,
                         deterministic=True,
                         logger=logger,
                         sync_batchnorm=True,
                         checkpoint_callback=checkpoint_callback,
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         limit_val_batches=10)

    return model, trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--export', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.run.visible_devices
    seed = config.run.seed or random.randint(1, 10000)
    seed_everything(seed)
    print('Using manual seed: {}'.format(seed))
    print('Config: ', config)

    model, trainer = build_trainer(config, seed, args)

    if args.train:
        trainer.fit(model)

    elif args.export:
        model.export_models('./')
