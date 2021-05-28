import os
import argparse

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from data.datasets import DataFoxNews, DataTwitterDavidson, ALL_DATASETS, BalancedSampler
from data_utils import TakeTurnLoader
from knn import KNNBaseline
import wandb
from pytorch_lightning.loggers import WandbLogger  # newline 1
import sklearn.metrics as mtr
from protomaml.data_utils import create_dataloader, prepare_batch, generate_tasks_from_dataset


def eval(args):

    # dataloaders
    pl.seed_everything(args.seed)
    wandb.init(project='baseline-eval', entity='atcs-project', config=args)

    # create dataloaders
    print("Evaluaing on:")
    datasets = []
    for dataset in args.datasets:
        assert dataset in ALL_DATASETS
        print("...", dataset)
        datasets.append(ALL_DATASETS[dataset]())

    n_support = args.n_support if args.n_support > 1 else 5
    tasks_per_dataset = [generate_tasks_from_dataset(d, support_examples=n_support,
                                         query_examples=args.n_query,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                        support_sampler=BalancedSampler,
                                        query_sampler=None,
                                         num_workers=4) for d in datasets]

    device = torch.device(args.device)
    dataset_results = []
    for i, tasks in enumerate(tasks_per_dataset):
        print("Evaluating dataset", i)
        # load the model


        model = KNNBaseline(**vars(args), clfs_spec=[datasets[i].n_classes]).to(device)
        state_dict = torch.load(args.model_path, map_location=device)

        # remove task specific weights
        for key in list(state_dict["state_dict"].keys()):
            if key.startswith("classifiers."):
                del state_dict["state_dict"][key]

        # overwrite task specific initialization
        for key in list(model.state_dict().keys()):
            if key.startswith("classifiers."):
                state_dict["state_dict"][key] = model.state_dict()[key]

        model.load_state_dict(state_dict["state_dict"])
        for_da_mean = []



        tasks = tasks[:10]
        for j, task in enumerate(tasks):
            seen_in_support = set()
            print("Loading model for task", j)
            model.reset_classifiers()
            model.classifiers.to(device)

            optimizer = torch.optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, model.encoder.bert.parameters())},
                 {'params': filter(lambda p: p.requires_grad, model.shared_mlp.parameters())},
                 {'params': filter(lambda p: p.requires_grad, model.classifiers.parameters()),
                  'lr': args.lr * 100}],
                lr=args.lr, weight_decay=args.weight_decay)
            # finetune on the support set
            print("Finetuning on support set")
            model.train()
            print("Number of support batches", len(task.support_loader))
            for k, data in enumerate(task.support_loader):

                if k >= args.n_support:
                    print("NOT TOO MANY BATCHES MY FIREND", k)
                    break

                batch_x, labels = data
                batch_x = [b.to(device) for b in batch_x]
                labels = labels.to(device)

                for p, sample in enumerate(batch_x[0]):
                    sample = tuple(list(sample.cpu().detach().numpy()[torch.nonzero(batch_x[1][p].cpu()).numpy().squeeze()]))
                    if not (sample in seen_in_support):
                        seen_in_support.add(sample)
                    else:
                        print("Sample already seen in support", k)

                # BATCH_x = [text, mask]
                x = model.forward(batch_x, classifier_idx=0)  # B x n_classes

                if x.shape[0] <= 1:
                    continue

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
                wandb.log({f"loss_{i}": loss, f"support_acc_{i}": accuracy, f"support_bal_acc_{i}": balanced_accuracy})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluate on the query set
            print("Evaluating on query set")
            model.eval()
            all_preds = []
            all_labels = []

            seen_in_query = set()

            with torch.no_grad():
                for l, data in enumerate(task.query_loader):
                    batch_x, labels = data
                    batch_x = [b.to(device) for b in batch_x]
                    labels = labels.to(device)

                    for p, sample in enumerate(batch_x[0]):
                        sample = tuple(list(sample.cpu().detach().numpy()[torch.nonzero(batch_x[1][p].cpu()).numpy().squeeze()]))
                        if not (sample in seen_in_query):
                            seen_in_query.add(sample)
                        else:
                            print("Sample already seen in query", k)

                        if sample in seen_in_support:
                            print("AAAAAAAAAH sample seen in support :(", k)


                    x = model.forward(batch_x, classifier_idx=0)  # B x n_classes

                    if x.shape[0] <= 1:
                        continue

                    if x.shape[-1] == 1:
                        # 2 classes
                        preds = torch.round(torch.sigmoid(x).squeeze()).detach().int().cpu()

                    else:
                        # >2 classes
                        preds = x.argmax(dim=1).detach().cpu()

                    all_preds += list(preds)
                    all_labels += list(labels.cpu())

            accuracy = mtr.accuracy_score(all_labels, all_preds)
            balanced_accuracy = mtr.balanced_accuracy_score(all_labels, all_preds)
            dataset_results.append({
                "dataset_name": task.task_name,
                "dataset": i,
                "task": j,
                "acc": accuracy,
                "bal_acc": balanced_accuracy,
                "f1_macro": mtr.f1_score(all_labels, all_preds, average='macro'),
                "confusion": mtr.confusion_matrix(all_labels, all_preds)
            })
            for_da_mean.append(dataset_results[-1])
        dataset_results.append({
            "dataset_name": for_da_mean[0]["dataset_name"],
            "dataset": for_da_mean[0]["dataset"],
            "task": "mean",
            "acc": sum([x["acc"] for x in for_da_mean]) / len(for_da_mean),
            "bal_acc": sum([x["bal_acc"] for x in for_da_mean]) / len(for_da_mean),
            "f1_macro": sum([x["f1_macro"] for x in for_da_mean]) / len(for_da_mean),
            "confusion": "nope"
        })
    res = pd.DataFrame(dataset_results)

    res.to_csv(f"evaluation_results_{'-'.join(args.datasets)}_{args.model_path.replace('/', '')}_{args.batch_size}_{args.n_support}_{args.n_query}.csv")


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default="twitter_davidson,foxnews", type=lambda x: x.split(","),
                        help='Names of datasets separated with commas.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Finetuning learning rate')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='Finetuning weight decay')
    parser.add_argument('--model_path', default="./models/foxnews.ckpt", type=str,
                        help='Path to the KNN model')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Device')
    parser.add_argument('--n_support', default=5, type=int,
                        help='Number of support examples used for finetuning')
    parser.add_argument('--n_query', default=5, type=int,
                        help='Number of support examples used for evaluating')
    args = parser.parse_args()
    eval(args)
