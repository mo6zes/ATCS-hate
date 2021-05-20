import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
from collections import defaultdict
from random import shuffle as randomshuffle
from torch.utils.data import Dataset, DataLoader, Subset, Sampler

from models.model_utils import create_tokenizer


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
    """ The metaloader is an iterator which returns Tasks"""
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
    texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
    return (texts['input_ids'], texts['attention_mask']), labels

def create_dataloader(dataset, batch_size=1, shuffle=None, num_workers=0,
                      collate_fn=prepare_batch, pin_memory=False, sampler=None):
    """Create a dataloader given a dataset. Small wrapper."""
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
	
def resample_tasks(tasks):
	rng = default_rng()
    resampled_tasks = []
    task_bucket = defaultdict(list)
    for task in tasks:
        task_bucket[task.task_name].append(task)
	num_task = torch.mean(torch.stack([torch.Tensor([len(task_bucket[i])]) for i in task_bucket.keys()]))
    for bucket in task_bucket.keys():
		if len(task_bucket[bucket]) >= num_task:
			sampled_tasks = rng.choice(task_bucket[bucket], size=num_task, replace=False)
			resampled_tasks.extend(sampled_tasks)
		else:
			multiple = num_task // len(task_bucket[bucket])
			residual = num_task - (multiple * len(task_bucket[bucket]))
			for i in range(multiple):
				resampled_tasks.extend(task_bucket[bucket])
			if residual > 0:
				sampled_tasks = rng.choice(task_bucket[bucket], size=residual, replace=False)
				resampled_tasks.extend(sampled_tasks)
    return resampled_tasks

def train_val_split(tasks, ratio=0.9, shuffle=True):
    if shuffle:
        randomshuffle(tasks)
    train = []
    val = []
    task_bucket = defaultdict(list)
    for task in tasks:
        task_bucket[task.task_name].append(task)
    for bucket in task_bucket.keys():
        split = int(np.round(len(task_bucket[bucket]) * ratio))
        train.extend(task_bucket[bucket][:split])
        val.extend(task_bucket[bucket][split:])
    return train, val
	
class BalancedTaskSampler(Sampler):
    """
    Sample from dataset in a balanced manner. No guarantee all the samples are seen during training.
    """
    def __init__(self, tasks):
        self.len_tasks = len(tasks)
        self.datasets_names = list(set([i.task_name for i in tasks]))

        self.indices = {}
        for i in self.datasets_names:
			lst = [index(j) for j in tasks if j.task_name==i]
            if len(lst) > 0:
				self.indices[i] = lst
            
        self.counts = {}
        for i in self.datasets_names:
            self.counts[i] = 0
        self.current_class = self.datasets_names[0]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.len_tasks:
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        chosen_class = self.get_class()
        class_indices = self.indices[chosen_class]
        chosen_index = np.random.choice(class_indices)
        self.counts[chosen_class] += 1
        return chosen_index

    def get_class(self):
        min_count = min([self.counts[i] for i in self.counts.keys()])
        min_classes = [int(min(self.counts, key=self.counts.get))]
        for i in self.num_classes:
            if self.counts[i] <= min_count and i not in min_classes:
                min_classes.append(i)
        chosen_class = np.random.choice(min_classes)
        return chosen_class

    def __len__(self):
        return self.len_tasks
