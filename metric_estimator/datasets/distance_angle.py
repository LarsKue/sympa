
from torch.utils.data import TensorDataset

from .stacked import StackedDataset
from .distance import DistanceDataset
from ..legacy import io


class DistanceAngleDataset(StackedDataset):
    def __init__(self, distances: DistanceDataset, angles: TensorDataset):
        super(DistanceAngleDataset, self).__init__([distances, angles])

    @classmethod
    def generate(cls, *, size, manifold, path, overwrite=False):
        path = io.prepare_path(path)
        io.handle_overwrite(path, overwrite)
        distances = DistanceDataset.generate(size=size, manifold=manifold, path=path, overwrite=overwrite)

        angles = ...

        angles = TensorDataset(angles)

        return cls(distances, angles)
