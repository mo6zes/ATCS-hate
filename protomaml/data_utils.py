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


def create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=prepare_batch, pin_memory=False):
    """Create a dataloader given a dataset. Small wrapper"""
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      pin_memory=pin_memory)


def generate_tasks_from_dataset(dataset, num_tasks=None, support_examples=100,
                                query_examples=100, **kwargs):
    """Slice in a dataset to return a list of Tasks."""
    # TODO I believe that in the meta dataset paper they had some rules on the number of sample in the support and query
    # for example that at least 1 of each class must be represented in the support, we might have to look into that

    interval = len(dataset) // (support_examples*kwargs['batch_size'] + query_examples*kwargs['batch_size'])
    if num_tasks and interval > num_tasks:
        interval = num_tasks
    elif num_tasks and interval < num_tasks:
        print(f"{num_tasks} tasks is to high, using {interval} tasks instead.")
    tasks = []
    for i in range(interval):
        start = i*(support_examples*kwargs['batch_size'] + query_examples*kwargs['batch_size'])
        support = Subset(dataset, range(start, start+(support_examples * kwargs['batch_size'])))
        start += support_examples*kwargs['batch_size']
        query = Subset(dataset, range(start, start+(query_examples*kwargs['batch_size'])))
        task = Task(task_name = dataset.task_name,
                    support_loader = create_dataloader(support, batch_size=kwargs['batch_size'],
                                                       shuffle=kwargs['shuffle'],
                                                       num_workers=kwargs['num_workers']),
                    query_loader = create_dataloader(query, batch_size=kwargs['batch_size'],
                                                       shuffle=kwargs['shuffle'],
                                                       num_workers=kwargs['num_workers']),
                    task_id = i,
                    n_classes = dataset.n_classes)
        tasks.append(task)
    return tasks
