import torch
from torch.utils.data import Dataset
import pickle
import copy

from ..legacy import io


class FileDataset(Dataset):
    """
    A dataset for large data sizes
    where every item is a torch.Tensor saved
    in a file named varname_index
    """

    def __init__(self, path, size: int, attribute_names: list[str]):
        self.path = io.prepare_path(path)
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

    def with_size(self, new_size):
        """
        Truncate the dataset
        @param new_size: Size of the truncated set
        @return: The truncated dataset
        """
        if new_size > self.size:
            raise ValueError(f"Cannot create {self.__class__} with size {new_size} from size {self.size}.")

        instance = copy.copy(self)

        instance.size = new_size

        return instance

    def transform(self, how):
        """
        Apply expensive transformations once across the dataset
        Number and order of tensors must match current dataset
        If you need a different transformation, construct a new dataset
        @param how: the transformation,
                    taking the current output tensors as arguments,
                    returning a tuple of new output tensors
        @return: None
        """
        for i, tensors in enumerate(self):
            tensors = how(tensors)

            for name, tensor in zip(self.attribute_names, tensors):
                torch.save(tensor, self.path / (name + "_" + str(i)))

    def save(self):
        """
        Save Metadata to the dataset directory
        @return: None
        """
        with open(self.path / "meta", "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load Metadata from the dataset directory
        @return: The dataset
        """
        path = io.prepare_path(path)
        with open(path / "meta", "rb") as f:
            return pickle.load(f)


