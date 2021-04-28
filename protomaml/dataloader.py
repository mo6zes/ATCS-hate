import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from BERT import create_tokenizer


class Task():
    # This task class hold two dataloader and some info about the task
    def __init__(self, task_name: str, support_loader: DataLoader, query_loader: DataLoader, task_id: int, n_classes: int):
        self.task_name = task_name
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.task_id = task_type
        self.n_classes = n_classes
        
class MetaDataloader(Dataset):
    # The metaloader is an iterator which randomly returns Tasks
    def __init__(self, tasks: list):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

def create_metaloader(tasks, batch_size=1, shuffle=False, num_workers=0, collate_fn=meta_collate_fn, pin_memory=False):
    dataset = MetaDataloader(tasks)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    return dataloader

class DataTwitterDavid(Dataset):
    # Test dataset class. All other dataset classes should include task_name and n_classes.
    def __init__(self, csv_file_dir:str):
        self.tweets = []
        self.label = []
        with open(csv_file_dir, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tweets.append(row['tweet'])
                self.label.append(torch.Tensor(int(row['class']), dtype=torch.long))
        
        assert len(self.tweets) == len(self.classes)
        self.n_classes = torch.numel(torch.unique(self.classes))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.class[idx]
    
def meta_collate_fn(batch):
    """Return the input without modification."""
    return batch

def prepare_batch(batch, tokenizer="bert-base-uncased"):
    """Transform the text and labels into tensors.
        The text is also tokized and padded automatically."""
    tokenizer = create_tokenizer(model_type=tokenizer)
    texts, labels = batch
    labels = torch.stack(labels)
    texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return (texts['input_ids'], texts['attention_mask']), labels

def create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=prepare_batch, pin_memory=False):
    """Create a dataloader given a dataset. Small wrapper"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)

def generate_tasks_from_dataset(dataset, num_tasks=None, support_examples=100, query_examples=100, **kwargs):
    """Slice in a dataset to return a list of Tasks."""
    interval = len(dataset) // (support_examples + query_examples)
    if num_tasks and interval > num_tasks:
        interval = num_tasks
    elif num_tasks and interval < num_tasks:
        print(f"{num_tasks} tasks is to high, using {interval} tasks instead.")
    tasks = []
    for i in range(interval):
        start = i*(support_examples + query_examples)
        support = Subset(dataset, range(start, support_examples))
        start += support_examples
        query = Subset(dataset, range(start, query_examples))
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