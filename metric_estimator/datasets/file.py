
import torch
from torch.utils.data import Dataset
import pickle


class FileDataset(Dataset):
    """
    A dataset for large data sizes
    where every item is a torch.Tensor saved
    in a file named varname_index
    """

    def __init__(self, path, size, attribute_names):
        self.path = path
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
        pickle.dump(self, self.path / "meta")

    @staticmethod
    def load(path):
        return pickle.load(path / "meta")
