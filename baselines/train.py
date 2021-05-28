import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from data.datasets import DataFoxNews, DataTwitterDavidson, ALL_DATASETS, BalancedSampler
from data_utils import TakeTurnLoader
from knn import KNNBaseline
import wandb
from pytorch_lightning.loggers import WandbLogger  # newline 1

from protomaml.data_utils import create_dataloader, prepare_batch

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
    print("training on:")
    datasets = []
    for dataset in args.datasets:
        assert dataset in ALL_DATASETS
        print("...", dataset)
        datasets.append(ALL_DATASETS[dataset]())
        print("....", datasets[-1].task_name)

    #TODO random splitting might produce imbalanced classes in splits
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for d in datasets:
        train_length = int(args.split_fractions[0] * len(d))
        val_length = int(args.split_fractions[1] * len(d))
        test_length = len(d) - train_length - val_length

        train_dataset, rest = torch.utils.data.random_split(d, [train_length, val_length + test_length])
        val_dataset, test_dataset = torch.utils.data.random_split(rest, [val_length, test_length])
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)


    train_dl = [create_dataloader(d,
                              batch_size=32,
                              shuffle=False,
                                num_workers=4,
                                  sampler=BalancedSampler(list(torch.tensor(d.dataset.labels)[d.indices])),
                              collate_fn=prepare_batch) for d in train_datasets]
    val_dls = [create_dataloader(d,
                              batch_size=32,
                              shuffle=False,
                                 num_workers=4,
                              collate_fn=prepare_batch) for d in val_datasets]
    # test_dl = TakeTurnLoader(test_datasets)

    print("Datasets")
    print("Training datasets: ", len(train_dl))
    print("Validation datasets: ", len(val_dls))

    # Create a PyTorch Lightning trainer
    callbacks = []
    modelcheckpoint = ModelCheckpoint(monitor='train_total_loss', mode='min', save_top_k=1,
                                      save_last=True, filename='{epoch}-{train_loss:.4f}-{train_acc:.3f}')
    callbacks.append(modelcheckpoint)
    # callbacks.append(LogCallback())
    if not args.progress_bar:
        callbacks.append(PrintCallback())
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    wandb_logger = WandbLogger(project='hate-baseline', entity='atcs-project', config=vars(args))
    # wandb.init(project='hate-baseline', entity='atcs-project', config=vars(args))
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         auto_select_gpus=torch.cuda.is_available(),
                         gpus=None if args.gpus == "None" else int(args.gpus),
                         max_epochs=args.epochs,
                         callbacks=callbacks,
                         auto_scale_batch_size='binsearch' if args.auto_batch else None,
                         auto_lr_find=True if args.auto_lr else False,
                         precision=args.precision,
                         resume_from_checkpoint=args.checkpoint_path,
                         gradient_clip_val=args.grad_clip,
                         benchmark=True if args.benchmark else False,
                         plugins=args.plugins,
                         logger=wandb_logger if args.wandb else None,
                         profiler=args.profiler if args.profiler else None,
                         multiple_trainloader_mode='max_size_cycle')
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
    with torch.autograd.set_detect_anomaly(True):
        # trainer.tune(model, train_dataloader=train_dl, val_dataloaders=val_dls)
    # TODO this might be problematic, not supplying a real dataloader, only something iterable.
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=train_dl + val_dls)
        print(modelcheckpoint.best_model_path)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser = KNNBaseline.add_model_specific_args(parser)

    # trainer hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training.')
    parser.add_argument('--split_fractions', default="0.9,0.1,0", type=lambda x: [float(f) for f in x.split(',')],
                        help='Split fractions e.g.: 0.9,0.1,0')
    parser.add_argument('--datasets', default="redditquian", type=lambda x: x.split(","),
                        help='Names of datasets separated with commas.')
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
