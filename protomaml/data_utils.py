import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    bucket_support_indices = [[] for _ in range(dataset.n_classes)]
    bucket_query_indices = [[] for _ in range(dataset.n_classes)]
    for i in range(dataset.n_classes):
        lst = torch.unbind(torch.nonzero(torch.stack(dataset.labels) == i).squeeze(-1))
        lst = [i.item() for i in lst]
        ratio = len(lst) // interval
        for j in range(interval):
            start = int(ratio * j)
            middle = int(ratio * (j + example_ratio))
            end = int(ratio * (j + 1))
            bucket_support_indices[i].append(lst[start:middle])
            bucket_query_indices[i].append(lst[middle:end])
        if len(lst) != end:
            middle = int(end + ((len(lst) - end)*example_ratio))
            bucket_support_indices[i].append(lst[end:middle])
            bucket_query_indices[i].append(lst[middle:len(lst)])

    for j in range(len(bucket_support_indices[0])):
        support_indices = []
        query_indices = []
        for i in range(dataset.n_classes):
            support_indices.extend(bucket_support_indices[i][j])
            query_indices.extend(bucket_query_indices[i][j])

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