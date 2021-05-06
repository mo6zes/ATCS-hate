import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import math
import bisect
from data.datasets import ALL_DATASETS
from models.model_utils import create_tokenizer
from protomaml.data_utils import create_dataloader, prepare_batch
from torch.utils.data.sampler import RandomSampler


class TakeTurnLoader:
    """
    Datasets take turns for returning a batch. Mode can be "oversample" or "undersample". In oversample mode all the data
    from the largest dataset will be returned once, smaller datasets will resample. In undersample mode the data for
    the smallest dataset will be returned once, some data in the largest data will not be returned in a batch.
    """

    def __init__(self, datasets, batch_size=32, shuffle=False, num_workers=0,
                 collate_fn=prepare_batch, pin_memory=False, mode="oversample"):
        self.dataloaders = [create_dataloader(d,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              pin_memory=pin_memory) for d in datasets]
        self.iterators = [iter(dl) for dl in self.dataloaders]

        dl_sizes = [len(d) for d in self. dataloaders]
        if mode == "oversample":
            self.max_batches_per_dataset = max(dl_sizes)
        elif mode == "undersample":
            self.max_batches_per_dataset = min(dl_sizes)
        else:
            raise NotImplementedError

    def __iter__(self):
        for n_batch in range(self.max_batches_per_dataset):
            for dataset_idx in range(len(self.dataloaders)):
                try:
                    batch = self.iterators[dataset_idx].__next__()
                except StopIteration:
                    self.iterators[dataset_idx] = self.dataloaders[dataset_idx].__iter__()
                    batch = self.iterators[dataset_idx].__next__()
                yield batch, dataset_idx


if __name__ == "__main__":
    datasets = [ALL_DATASETS['fox_news'](), ALL_DATASETS['twitter_davidson']()]

    dl = TakeTurnLoader(datasets)

    b = next(iter(dl))


