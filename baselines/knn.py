import torch.nn as nn
import pytorch_lightning as pl


from models.BERT import BERT
import torch
from models.model_utils import create_pretrained_transformer


class KNNBaseline(pl.LightningModule):

    def __init__(self, similarity='euclidean', transformer_model='bert-base-uncased', hidden_size=512,
                 output_size=512, clfs_spec=None, lr=1e-2,  weight_decay=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = BERT(transformer_model=transformer_model, hidden_size=hidden_size, output_size=output_size)
        self.encoder.unfreeze_module()

        assert clfs_spec, "Classifier specification is required."

        classifiers = []

        for n_classes in clfs_spec:
            assert n_classes > 1
            n_out = 1 if n_classes == 2 else n_classes
            classifiers.append(nn.Sequential(
                nn.Linear(output_size, n_out)
            ))

        self.classifiers = nn.ModuleList(modules=classifiers)

    def forward(self, x, classifier_idx=0):
        x = self.encoder(x)  # B x output_dim
        return self.classifiers[classifier_idx](x)  # B * n_classes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifiers.parameters()),
                                    lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, dataset_idx = batch
        # text is a tuple of (words, mask)
        text, labels = data

        x = self.forward(text, classifier_idx=dataset_idx)  # B x n_classes

        if x.shape[-1] == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x.squeeze(), labels.float())
            accuracy = (labels == torch.round(x.squeeze()).detach().int()).sum() / labels.size(0)
        else:
            loss = torch.nn.functional.cross_entropy(x, labels)
            accuracy = (labels == x.argmax(dim=1)).sum() / labels.size(0)

        self.log("train_dataset", dataset_idx)
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)

        return loss

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





