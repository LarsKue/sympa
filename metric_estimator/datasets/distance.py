
import torch
from torch.utils.data import TensorDataset

from .points import SiegelPointDataset
from .stacked import StackedDataset
from ..device import device


class DistanceDataset(StackedDataset):
    def __init__(self, points: SiegelPointDataset, distances: TensorDataset):
        super(DistanceDataset, self).__init__([points, distances])

    @classmethod
    def generate(cls, *, size, manifold, path, overwrite=False):
        points = SiegelPointDataset.generate(size=size, ndim=manifold.ndim, path=path, overwrite=overwrite)

        print(f"Calculating {size} distances...")
        distances = torch.Tensor([
            manifold.dist(z1, z2) for z1, z2 in points
        ]).to(device)

        distances = distances.unsqueeze(-1)

        distances = TensorDataset(distances)

        return cls(points, distances)
