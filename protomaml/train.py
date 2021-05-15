import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from protomaml.protomaml import ProtoMAML
import wandb

from .utils import generate_tasks
from .data_utils import create_metaloader, train_val_split
from data.datasets import DataTwitterDavidson, DataFoxNews, DeGilbertStormFront, QuianData, RezvanHarrassment, FountaDataset, TalkdownDataset, WikipediaDataset


class LogCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)
            
class MemoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache() 
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()  
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()  
            
class PrintCallback(pl.Callback):
    def __init__(self):
        super().__init__()
            
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        print(f"Epoch {trainer.current_epoch} finished.")

def train(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    pl.seed_everything(args.seed)
#     os.makedirs(args.log_dir, exist_ok=True)
    
    # create logger
    wandb_logger = WandbLogger(project='protomaml', entity='atcs-project', tags=['meta-learning', 'protomaml'], version=0, log_model=True, group="ProtoMAML")
    
    # create dataloaders
    tasks = generate_tasks(args, dataset_list=[DataFoxNews(), DeGilbertStormFront(), QuianData(),
                                               RezvanHarrassment(), FountaDataset(), WikipediaDataset()])
    train_tasks, val_tasks = train_val_split(tasks, shuffle=False)
    
    meta_train_loader = create_metaloader(train_tasks, batch_size=args.meta_batch_size, shuffle=True)
    meta_val_loader = create_metaloader(val_tasks, batch_size=args.meta_batch_size, shuffle=False)
    
    test_Twitter = generate_tasks(args, dataset_list=[DataTwitterDavidson()], sampler=None)
    test_QuianReddit = generate_tasks(args, dataset_list=[QuianData("./raw_datasets/redditQuian.csv")], sampler=None)
    test_Talkdown = generate_tasks(args, dataset_list=[TalkdownDataset()], sampler=None)
    
    meta_test_Twitter_loader = create_metaloader(test_Twitter, batch_size=args.meta_batch_size, shuffle=False)
    meta_test_QuianReddit_loader = create_metaloader(test_QuianReddit, batch_size=args.meta_batch_size, shuffle=False)
    meta_test_Talkdown_loader = create_metaloader(test_Talkdown, batch_size=args.meta_batch_size, shuffle=False)
    
    print(f"Training using {len(train_tasks)} tasks, evaluating with {len(val_tasks)} tasks.")

    # Create a PyTorch Lightning trainer
    callbacks = []
    callbacks.append(MemoryCallback())
    modelcheckpoint = ModelCheckpoint(monitor='val_query_f1', mode='max', save_top_k=1,
                                      save_last=True, filename='{epoch}-{train_query_loss:.3f}-{train_query_acc:.3f}-{train_query_f1:.3f}-{val_query_loss:.3f}-{val_query_acc:.3f}-{val_query_f1:.3f}')
    callbacks.append(modelcheckpoint)
    callbacks.append(LearningRateMonitor())
    try:
        wandb_logger
    except Exception:
        callbacks.append(LogCallback())
    callbacks.append(EarlyStopping(monitor='val_query_f1', mode='max', patience=3))
    if not args.progress_bar:
        callbacks.append(PrintCallback())
        
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         auto_select_gpus=True,
                         gpus=None if args.gpus == "None" else int(args.gpus),
                         max_epochs=args.epochs,
                         callbacks=callbacks,
                         auto_scale_batch_size='binsearch' if args.auto_batch else None,
                         auto_lr_find=True if args.auto_lr else False,
                         precision=args.precision,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         limit_train_batches=args.train_limit,
                         limit_val_batches=args.val_limit,
                         limit_test_batches=args.test_limit,
                         accumulate_grad_batches=args.grad_batch,
                         resume_from_checkpoint=args.checkpoint_path,
                         gradient_clip_val=args.grad_clip,
                         benchmark=True if args.benchmark else False,
                         plugins=args.plugins,
                         profiler=args.profiler if args.profiler else None,
                         logger=wandb_logger)
#     trainer.logger._default_hp_metric = None
#     trainer.logger._log_graph = False
    
    # Create model
    dict_args = vars(args)
    model = ProtoMAML(**dict_args)
        
#     if not args.progress_bar:
#         print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
#               "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
#               "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.set_detect_anomaly(True):
    if not args.test_model:
        trainer.tune(model, train_dataloader=meta_train_loader, val_dataloaders=meta_val_loader)
        trainer.fit(model, train_dataloader=meta_train_loader, val_dataloaders=meta_val_loader)
        print(modelcheckpoint.best_model_path)
    else:
        trainer.test(model, test_dataloaders=[meta_test_Twitter_loader, meta_test_QuianReddit_loader, meta_test_Talkdown_loader])


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser = ProtoMAML.add_model_specific_args(parser)
    
    # trainer hyperparameters
    parser.add_argument('--epochs', default=25, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training.')
    parser.add_argument('--meta_batch_size', default=16, type=int,
                        help='Amount of tasks.')
    
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for the tasks.')
    
    parser.add_argument('--query_examples', default=100, type=int,
                        help='Number of batches in the query loader.')
    
    parser.add_argument('--precision', default=32, type=int,
                        choices=[16, 32],
                        help='At what precision the model should train.')
    parser.add_argument('--grad_batch', default=1, type=int,
                        help='Accumulate gradient to simulate larger batch sizes.')
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
    
    parser.add_argument('--train_limit', default=1.0, type=float,
                        help='Percentage of data to train on.')
    parser.add_argument('--val_limit', default=1.0, type=float,
                        help='Percentage of data to validate with.')
    parser.add_argument('--test_limit', default=1.0, type=float,
                        help='Percentage of data to test with.')
    
    parser.add_argument('--test_model', action='store_true',
                        help='Test the current model in k-shot fashion.')
    
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
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