import torch.nn as nn
import pytorch_lightning as pl
import sklearn.metrics as mtr
from torch.optim.lr_scheduler import LambdaLR

from models.BERT import BERT
import torch
import matplotlib.pyplot as plt
from models.model_utils import create_pretrained_transformer
import pandas as pd
import seaborn as sns
import wandb
import transformers

class KNNBaseline(pl.LightningModule):

    def __init__(self, similarity='euclidean', transformer_model='bert-base-uncased', hidden_size=512,
                 output_size=512, clfs_spec=None, lr=5e-5,  weight_decay=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        print("loading bert")
        self.encoder = BERT(transformer_model=transformer_model, hidden_size=hidden_size, output_size=0)
        self.encoder.unfreeze_attention_layer([6, 7, 8, 9, 10, 11])
        self.shared_mlp = nn.Sequential(
                      nn.Linear(self.encoder.bert.config.hidden_size, hidden_size),
                      nn.ReLU()
                    )

        assert clfs_spec, "Classifier specification is required."

        classifiers = []

        for n_classes in clfs_spec:
            assert n_classes > 1
            n_out = 1 if n_classes == 2 else n_classes
            classifiers.append(nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, n_out)
            ))

        self.classifiers = nn.ModuleList(modules=classifiers)

    def forward(self, x, classifier_idx=0):
        x = self.encoder(x)  # B x output_dim
        x = self.shared_mlp(x)
        return self.classifiers[classifier_idx](x)  # B * n_classes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.encoder.bert.parameters())},
             {'params': filter(lambda p: p.requires_grad, self.shared_mlp.parameters()), 'lr': self.hparams.lr * 100},
             {'params': filter(lambda p: p.requires_grad, self.classifiers.parameters()), 'lr': self.hparams.lr * 100}],
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warm_up_steps = self.hparams.lr_warmup_steps

        def lr_lambda(epoch: int):
            epoch += 1
            if epoch < warm_up_steps:
                return float(epoch) / float(max(1.0, warm_up_steps))
            return 1.0

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': False
            }]

    def training_step(self, batches, batch_idx):
        losses = []
        for dataset_idx, data in enumerate(batches):
            # text is a tuple of (words, mask)
            text, labels = data

            # print(labels)

            x = self.forward(text, classifier_idx=dataset_idx)  # B x n_classes
            if x.shape[-1] == 1:
                # 2 classes
                loss = torch.nn.functional.binary_cross_entropy_with_logits(x.squeeze(), labels.float())
                preds = torch.round(torch.sigmoid(x).squeeze()).detach().int().cpu()

            else:
                # >2 classes
                loss = torch.nn.functional.cross_entropy(x, labels)
                preds = x.argmax(dim=1).detach().cpu()

            # print(x)
            # print(preds)
            # print(labels)
            accuracy = mtr.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            balanced_accuracy = mtr.balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            # print(accuracy)
            # print(balanced_accuracy)
            self.log(f"train_{dataset_idx}_dataset", dataset_idx)
            self.log(f"train_{dataset_idx}_loss", loss)
            self.log(f"train_{dataset_idx}_acc", accuracy)
            self.log(f"train_{dataset_idx}_bal_acc", balanced_accuracy)

            losses.append(loss)
        total_loss = sum(losses)
        self.log(f"train_total_loss", total_loss)
        return total_loss

    def validation_step(self, data, batch_idx, dataset_idx=None):
        # text is a tuple of (words, mask)
        if dataset_idx is None:
            dataset_idx=0

        text, labels = data

        x = self.forward(text, classifier_idx=dataset_idx % len(self.classifiers))  # B x n_classes

        if x.shape[-1] == 1:
            # 2 classes
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x.squeeze(), labels.float())
            preds = torch.round(torch.sigmoid(x).squeeze()).detach().int().cpu()

        else:
            # >2 classes
            loss = torch.nn.functional.cross_entropy(x, labels)
            preds = x.argmax(dim=1).detach().cpu()

        accuracy = mtr.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        balanced_accuracy = mtr.balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = mtr.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

        self.log(f"val_{dataset_idx}_f1", f1)
        self.log(f"val_{dataset_idx}_loss", loss)
        self.log(f"val_{dataset_idx}_acc", accuracy)
        self.log(f"val_{dataset_idx}_bal_acc", balanced_accuracy)

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_epoch_end(self, all_outputs):
        if not isinstance(all_outputs[0], list):
            all_outputs = [all_outputs]

        for i, outputs in enumerate(all_outputs):
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            labels = torch.cat([tmp['labels'] for tmp in outputs])

            wandb.log({f"val_conf_mat_{i}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels.cpu().numpy(),
                preds=preds.numpy())}, commit=False)

    def reset_classifiers(self):
        classifiers = []
        for n_classes in self.hparams.clfs_spec:
            assert n_classes > 1
            n_out = 1 if n_classes == 2 else n_classes
            classifiers.append(nn.Sequential(
                nn.Linear(self.hparams.hidden_size, self.hparams.output_size),
                nn.ReLU(),
                nn.Linear(self.hparams.output_size, n_out)
            ))

        self.classifiers = nn.ModuleList(modules=classifiers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("KNNBaseline")

        # model parameters
        parser.add_argument('--model', default='bert-base-uncased', type=str,
                            help='Which model to train.')
        parser.add_argument('--hidden_size', default=512, type=int,
                            help='Hidden size of the MLP on top of the transformer.')
        parser.add_argument('--output_size', default=512, type=int,
                            help='Output size of the MLP on top of the transformer.')

        # Optimizer hyperparameters
        parser.add_argument('--lr', default=1e-3, type=float,
                            help='Learning rate to use.')
        parser.add_argument('--lr_warmup_steps', default=4, type=int,
                            help='Number of warmup steps')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='Weight decay.')

        return parent_parser


if __name__ == "__main__":
    from data.datasets import ALL_DATASETS
    from baselines.data_utils import TakeTurnLoader

    datasets = [ALL_DATASETS['fox_news'](), ALL_DATASETS['twitter_davidson']()]

    dl = TakeTurnLoader(datasets)

    k = KNNBaseline(clfs_spec=[2, 3])

    batch = next(iter(dl))
    data, dataset_idx = batch
    # text is a tuple of (words, mask)
    text, labels = data
    x = k.encoder(text)
    x = k.classifiers[dataset_idx](x)





