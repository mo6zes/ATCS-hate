from .data_utils import generate_tasks_from_dataset
from data.datasets import DataTwitterDavidson, DataFoxNews, DeGilbertStormFront, QuianData, RezvanHarrassment, FountaDataset, BalancedSampler

def generate_tasks(args, dataset_list=[DataTwitterDavidson(), DataFoxNews(), QuianData(), QuianData("./raw_datasets/redditQuian.csv"), RezvanHarrassment(), FountaDataset()], sampler=BalancedSampler):
    tasks = []
    for dataset in dataset_list:
        tasks_set = generate_tasks_from_dataset(dataset, support_examples=args.inner_updates,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers,
                                                sampler=None)
        tasks.extend(tasks_set)
    return tasks