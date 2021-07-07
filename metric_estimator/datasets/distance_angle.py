
from torch.utils.data import TensorDataset

from .io import prepare_path
from .stacked import StackedDataset
from .distance import DistanceDataset


class DistanceAngleDataset(StackedDataset):
    def __init__(self, distances: DistanceDataset, angles: TensorDataset):
        super(DistanceAngleDataset, self).__init__([distances, angles])

    @classmethod
    def generate(cls, *, size, manifold, path, overwrite=False):
        path = prepare_path(path, overwrite)
        distances = DistanceDataset.generate(size=size, manifold=manifold, path=path, overwrite=overwrite)

        angles = ...

        angles = TensorDataset(angles)

        return cls(distances, angles)
