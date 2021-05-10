import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from data.datasets import DataFoxNews, DataTwitterDavidson
from data_utils import TakeTurnLoader
from knn import KNNBaseline
import wandb
from pytorch_lightning.loggers import WandbLogger  # newline 1

class LogCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)


class PrintCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        print(f"Epoch {trainer.current_epoch} finished.")


def train(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    pl.seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # create dataloaders
    datasets = [DataFoxNews(), DataTwitterDavidson()]

    # TODO split in training, val test based. Maybe leave out a task? Keep in mind that datasets have different sizes
    dl = TakeTurnLoader(datasets)

    # Create a PyTorch Lightning trainer
    callbacks = []
    modelcheckpoint = ModelCheckpoint(monitor='train_acc', mode='max', save_top_k=1,
                                      save_last=True, filename='{epoch}-{train_loss:.4f}-{train_acc:.3f}')
    callbacks.append(modelcheckpoint)
    callbacks.append(LogCallback())
    if not args.progress_bar:
        callbacks.append(PrintCallback())

    wandb_logger = WandbLogger(project='hate-baseline', entity='atcs-project')

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         auto_select_gpus=torch.cuda.is_available(),
                         gpus=None if args.gpus == "None" else int(args.gpus),
                         max_epochs=args.epochs,
                         callbacks=callbacks,
                         auto_scale_batch_size='binsearch' if args.auto_batch else None,
                         auto_lr_find=True if args.auto_lr else False,
                         precision=args.precision,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         resume_from_checkpoint=args.checkpoint_path,
                         gradient_clip_val=args.grad_clip,
                         benchmark=True if args.benchmark else False,
                         plugins=args.plugins,
                         logger=wandb_logger if args.wandb else None,
                         profiler=args.profiler if args.profiler else None)
    trainer.logger._default_hp_metric = None
    trainer.logger._log_graph = False

    # Create model
    dict_args = vars(args)
    model = KNNBaseline(**dict_args, clfs_spec=[d.n_classes for d in datasets])

    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.set_detect_anomaly(True):
    # TODO this might be problematic, not supplying a real dataloader, only something iterable.
    trainer.tune(model, train_dataloader=dl)
    trainer.fit(model, train_dataloader=dl)
    print(modelcheckpoint.best_model_path)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser = KNNBaseline.add_model_specific_args(parser)

    # trainer hyperparameters
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training.')

    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for the tasks.')

    parser.add_argument('--precision', default=32, type=int,
                        choices=[16, 32],
                        help='At what precision the model should train.')
    parser.add_argument('--grad_clip', default=0, type=float,
                        help='Clip the gradient norm.')
    parser.add_argument('--plugins', default=None, type=str,
                        help='Modify the multi-gpu training path. See docs lightning docs for details.')

    parser.add_argument('--gpus', default=1, type=str,
                        help='Which gpus to use.')

    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Continue training from this checkpoint.')

    # other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--log_dir', default='logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')

    parser.add_argument('--wandb', default=True, type=bool,
                        help='Use wandb to log runs')

    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. ' + \
                             'Not to be used in conjuction with SLURM jobs.')
    parser.add_argument('--auto_lr', action='store_true',
                        help='When used tries to automatically set an appropriate learning rate.')
    parser.add_argument('--auto_batch', action='store_true',
                        help='When used tries to automatically set an appropriate batch size.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enables cudnn auto-tuner.')

    parser.add_argument('--profiler', default=None, type=str,
                        choices=['simple', 'advanced', 'pytorch'],
                        help='Code profiler.')

    args = parser.parse_args()
    train(args)
