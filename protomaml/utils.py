import torch
from .data_utils import generate_tasks_from_dataset
from data.datasets import DataTwitterDavidson, DataFoxNews, DeGilbertStormFront, QuianData, RezvanHarrassment, FountaDataset, TalkdownDataset, WikipediaDataset, BalancedSampler

def generate_tasks(args, dataset_list=[DataTwitterDavidson(), DataFoxNews(), DeGilbertStormFront(),
                                       QuianData(), QuianData("./raw_datasets/redditQuian.csv"),
                                       RezvanHarrassment(), FountaDataset(), TalkdownDataset(),
                                       WikipediaDataset()],
                                       sampler=BalancedSampler):
    tasks = []
    for dataset in dataset_list:
        tasks_set = generate_tasks_from_dataset(dataset, support_examples=args.inner_updates,
                                                query_examples=args.query_examples, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers,
                                                sampler=None)
        tasks.extend(tasks_set)
    return tasks

def step(self, loss, params=None, override=None, grad_callback=None, retain_graph=False, **kwargs):
    """Override the higher library."""
    # Deal with override
    if override is not None:
        self._apply_override(override)

    if self._fmodel is None or self._fmodel.fast_params is None:
        if params is None:
            raise ValueError(
                "params kwarg must be passed to step if the differentiable "
                "optimizer doesn't have a view on a patched model with "
                "params."
            )
    else:
        params = self._fmodel.fast_params if params is None else params

    params = list(params)

    # This allows us to gracefully deal with cases where params are frozen.
    grad_targets = [
        p if p.requires_grad else torch.tensor([], requires_grad=True)
        for p in params
    ]

    all_grads = torch.autograd.grad(
        loss,
        grad_targets,
        create_graph=self._track_higher_grads,
        allow_unused=True,
        retain_graph=retain_graph
    )

    if grad_callback is not None:
        all_grads = grad_callback(all_grads)
    elif self._grad_callback is not None:
        all_grads = self._grad_callback(all_grads)

    grouped_grads = []
    for group, mapping in zip(self.param_groups, self._group_to_param_list):
        grads = []
        for i, index in enumerate(mapping):
            group['params'][i] = params[index]
            grads.append(all_grads[index])
        grouped_grads.append(grads)

    self._update(grouped_grads)

    new_params = params[:]
    for group, mapping in zip(self.param_groups, self._group_to_param_list):
        for p, index in zip(group['params'], mapping):
            if self._track_higher_grads:
                new_params[index] = p
            else:
                new_params[index] = p.detach().requires_grad_()

    if self._fmodel is not None:
        self._fmodel.update_params(new_params)

    return new_params