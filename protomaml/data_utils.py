import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
from torch.utils.data import Dataset, DataLoader, Subset

from models.model_utils import create_tokenizer

import time

class Task():
    """ This task class hold two dataloader and some info about the task"""
    def __init__(self, task_name: str, support_loader: DataLoader,
                 query_loader: DataLoader, task_id: int, n_classes: int):
        self.task_name = task_name
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.task_id = task_id
        self.n_classes = n_classes

class MetaDataloader(Dataset):
    """ The metaloader is an iterator which randomly returns Tasks"""
    def __init__(self, tasks: list):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

def meta_collate_fn(batch):
    """Return the input without modification."""
    return batch

def create_metaloader(tasks, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=meta_collate_fn, pin_memory=False):
    dataset = MetaDataloader(tasks)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    return dataloader

def prepare_batch(batch, tokenizer=create_tokenizer()):
    """Transform the text and labels into tensors.
        The text is also tokenized and padded automatically."""
    texts = [i[0] for i in batch]
    labels = torch.stack([i[-1] for i in batch])
    texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return (texts['input_ids'], texts['attention_mask']), labels

def create_dataloader(dataset, batch_size=1, shuffle=None, num_workers=0,
                      collate_fn=prepare_batch, pin_memory=False, sampler=None):
    """Create a dataloader given a dataset. Small wrapper"""
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      pin_memory=pin_memory,
                      sampler=sampler)

def generate_tasks_from_dataset(dataset, num_tasks=None, support_examples=100,
                                query_examples=10, sampler=None, **kwargs):
    """Slice in a dataset to return a list of Tasks."""
    tasks = []

    support_examples *= kwargs['batch_size']
    query_examples *= kwargs['batch_size']
    interval = len(dataset) // (support_examples + query_examples)
    example_ratio = support_examples / (support_examples + query_examples)
    rng = default_rng()
    class_bucket = []
    samples_per_class = []

    for i in range(dataset.n_classes):
        lst = torch.unbind(torch.nonzero(torch.stack(dataset.labels) == i).squeeze(-1))
        lst = [i.item() for i in lst]
        class_bucket.append(lst)
        samples_per_class.append(len(lst) // interval)

    # oversample
    for i in range(len(class_bucket)):
        residual = len(class_bucket[i]) - (interval * samples_per_class[i])
        size = abs(samples_per_class[i] - residual)
        class_samples = rng.choice(class_bucket[i], size=size, replace=False)
        class_bucket[i].extend(class_samples)

    # create sets
    support_bucket = []
    query_bucket = []
    for i in range(interval+1):
        support_set = []
        query_set = []
        for j in range(dataset.n_classes):
            samples = class_bucket[j]
            ratio = samples_per_class[j]
            start = int(np.rint(ratio * i))
            middle = int(np.rint(ratio * (i + example_ratio)))
            end = int(np.rint(ratio * (i + 1)))
            support_set.extend(samples[start:middle])
            query_set.extend(samples[middle:end])
        support_bucket.append(support_set)
        query_bucket.append(query_set)
        
    for support_indices, query_indices in zip(support_bucket, query_bucket):
        
        support_set = Subset(dataset, support_indices)
        query_set = Subset(dataset, query_indices)
        
        if sampler:
            support_loader = create_dataloader(support_set,
                                               batch_size=kwargs['batch_size'],
                                               num_workers=kwargs['num_workers'],
                                               sampler=sampler([dataset.labels[i] for i in support_indices]))
            query_loader = create_dataloader(query_set,
                                             batch_size=kwargs['batch_size'],
                                             num_workers=kwargs['num_workers'],
                                             sampler=sampler([dataset.labels[i] for i in query_indices]))
        else:
            support_loader = create_dataloader(support_set,
                                               batch_size=kwargs['batch_size'],
                                               shuffle=kwargs['shuffle'],
                                               num_workers=kwargs['num_workers'])
            query_loader = create_dataloader(query_set,
                                             batch_size=kwargs['batch_size'],
                                             shuffle=kwargs['shuffle'],
                                             num_workers=kwargs['num_workers'])
            
        task = Task(task_name = dataset.task_name,
                    support_loader = support_loader,
                    query_loader = query_loader,
                    task_id = i,
                    n_classes = dataset.n_classes)
        tasks.append(task)
    return tasks