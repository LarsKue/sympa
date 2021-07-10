import torch
from torch.utils.data import Dataset
import pickle

from .io import prepare_path


class FileDataset(Dataset):
    """
    A dataset for large data sizes
    where every item is a torch.Tensor saved
    in a file named varname_index
    """

    def __init__(self, path, size: int, attribute_names: list[str]):
        self.path = prepare_path(path)
        self.size = size
        self.attribute_names = attribute_names

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        item = self.clip_index(item)

        tensors = [
            torch.load(self.path / (name + "_" + str(item)))
            for name in self.attribute_names
        ]

        return tuple(tensors)

    def clip_index(self, index):
        size = len(self)
        if abs(index) >= size:
            raise IndexError(f"Index {index} is out of range for {type(self).__name__} with size {size}")
        if index < 0:
            index = size + index

        return index

    def save(self):
        with open(self.path / "meta", "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        path = prepare_path(path)
        with open(path / "meta", "rb") as f:
            return pickle.load(f)

    def transform(self, how, new_attribute_names: list[str]):
        """
        Apply expensive transformations once across the dataset
        @param how: the transformation,
                    taking the current output tensors as arguments,
                    returning the new output tensors
        @param new_attribute_names:
        @return:
        """
        for i, tensors in enumerate(self):
            tensors = how(tensors)

            if torch.is_tensor(tensors):
                # just a single tensor
                name = new_attribute_names[0]
                torch.save(tensors, self.path / (name + "_" + str(i)))
            else:
                # multiple tensors
                for name, t in zip(new_attribute_names, tensors):
                    torch.save(t, self.path / (name + "_" + str(i)))

        self.attribute_names = new_attribute_names
