
from torch.utils.data import Dataset
from typing import Callable


class TransformDataset(Dataset):
    # TODO: this can be done better, e.g. with dynamic inheritance or something like that

    def __init__(self, untransformed: Dataset, transform: Callable):
        self.untransformed = untransformed
        self.transform = transform

    def __getitem__(self, item):
        response = self.untransformed[item]
        return self.transform(response)

    def __len__(self):
        return len(self.untransformed)
