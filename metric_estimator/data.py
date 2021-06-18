
import torch
from torch.utils.data import Dataset
import pathlib as pl
import shutil
from deprecated import deprecated

from .device import device
from .utils import symmetricize, flat_batched_complex_triu


class MetricDistanceSet(Dataset):
    """
    Metric Distance Data Set
    Suitable for high dimensional data, as it
    saves input tensors in individual files
    """
    def __init__(self, path, labels):
        self.labels = labels
        self.path = path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        n = len(self)
        if abs(item) >= n:
            raise IndexError(f"Index {item} is out of range for MetricDistanceSet with length {n}")
        if item < 0:
            item = n - item

        z = torch.load(self.path / str(item))
        y = self.labels[item].unsqueeze(-1)

        return z, y

    @classmethod
    def load(cls, path):
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        labels = torch.load(path / "labels")

        return cls(path, labels)

    @classmethod
    def generate(cls, n, manifold, path, overwrite=False):
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        if not overwrite and path.exists():
            raise FileExistsError(f"Cannot overwrite {path}.")
        if overwrite and path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True)

        dists = []

        for i in range(n):
            z1 = generate_points(1, ndim=manifold.ndim)
            z2 = generate_points(1, ndim=manifold.ndim)
            dist = manifold.dist(z1, z2)

            zz1 = flat_batched_complex_triu(z1)
            zz2 = flat_batched_complex_triu(z2)

            z = torch.cat((zz1, zz2), dim=-1).squeeze()

            torch.save(z, path / str(i))
            dists.append(dist.item())

        dists = torch.Tensor(dists)
        torch.save(dists, path / "labels")
        return cls(path, dists)


@deprecated(version="0.0.1", reason="Use MetricDistanceSet instead")
class SimpleMetricDistanceSet(Dataset):
    """
    Simple Version of the MetricDistanceSet
    where the whole dataset fits into memory
    """
    def __init__(self, z1, z2, dist, z=None):
        if z is None:
            zz1 = flat_batched_complex_triu(z1)
            zz2 = flat_batched_complex_triu(z2)

            z = torch.cat((zz1, zz2), dim=-1)

        self.z1 = z1
        self.z2 = z2
        self.dist = dist
        self.z = z

    def __getitem__(self, item):
        return self.z[item], self.dist[item]

    def __len__(self):
        return len(self.dist)

    def save(self, path):
        """
        Save the dataset to a directory
        """
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        if path.exists():
            raise FileExistsError(f"Cannot overwrite {path}.")

        path.mkdir(parents=True, exist_ok=False)

        torch.save(self.z1, path / "z1")
        torch.save(self.z2, path / "z2")
        torch.save(self.z, path / "z")
        torch.save(self.dist, path / "dist")

    @classmethod
    def load(cls, path):
        """
        Load a dataset from a directory
        """
        z1 = torch.load(path / "z1")
        z2 = torch.load(path / "z2")
        z = torch.load(path / "z")
        dist = torch.load(path / "dist")

        return cls(z1, z2, dist, z=z)

    @classmethod
    def generate(cls, n, manifold):
        """
        Generate a new dataset
        """
        z1 = generate_points(n, ndim=manifold.ndim)
        z2 = generate_points(n, ndim=manifold.ndim)
        dist = manifold.dist(z1, z2).unsqueeze(-1)

        z1 = z1.cpu()
        z2 = z2.cpu()
        dist = dist.cpu()

        return cls(z1, z2, dist)


def generate_points(n, *, ndim):
    """
    Generate Points in a Siegel Space of given dimension
    Points are given by a Standard Normal Distribution

    """

    shape = (n, ndim, ndim)

    real = torch.randn(shape, device=device)
    imag = torch.randn(shape, device=device)

    # symmetricize
    real = symmetricize(real)
    imag = symmetricize(imag)

    # TODO: Confirm with Federico that this is fine
    #  since the distribution is affected by the dominant diagonal
    # positive definiteness
    evs = torch.symeig(imag)
    e = -evs.eigenvalues[:, 0] + 1
    e = e.unsqueeze(-1).unsqueeze(-1)

    diag = e * torch.eye(ndim, device=device)
    imag.add_(diag)

    return torch.stack((real, imag), dim=1)








