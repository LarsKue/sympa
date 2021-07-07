
import torch
from torch.utils.data import Dataset

from typing import Iterable


class StackedDataset(Dataset):
    """
    Stacks Two or more Datasets
    Example:
        >>> ds1, ds2 = ...
        >>> ds1[0]
        tensor([0]), tensor([1])
        >>> ds2[0]
        tensor([-1]), tensor([-2])
        >>> stacked_ds = StackedDataset([ds1, ds2])
        >>> stacked_ds[0]
        tensor([0]), tensor([1]), tensor([-1]), tensor([-2])
    """
    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = list(datasets)

    def __getitem__(self, item):
        responses = []
        for ds in self.datasets:
            response = ds[item]
            if isinstance(response, Iterable) and not torch.is_tensor(response):
                responses.extend(response)
            else:
                responses.append(response)

        return tuple(responses)

    def __len__(self):
        return len(self.datasets[0])
