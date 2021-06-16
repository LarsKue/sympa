import torch
from torch.utils.data import IterableDataset

from .device import device
from .utils import symmetricize


class MetricDistanceSet(IterableDataset):
    def __init__(self, z1, z2, dist):
        self.z1 = z1
        self.z2 = z2
        self.dist = dist

    def __iter__(self):
        return zip(self.z1, self.z2, self.dist)

    def __getitem__(self, item):
        return self.z1[item], self.z2[item], self.dist[item]

    def __len__(self):
        return len(self.dist)

    @classmethod
    def from_path(cls, path):
        z1 = torch.load(path / "z1")
        z2 = torch.load(path / "z2")
        dist = torch.load(path / "dist")

        return cls(z1, z2, dist)

    @classmethod
    def generate(cls, n, *, manifold, path):
        if path.exists():
            raise FileExistsError(f"Cannot overwrite {path}.")

        path.mkdir(parents=True, exist_ok=False)

        z1 = generate_points(n, ndim=manifold.ndim)
        z2 = generate_points(n, ndim=manifold.ndim)
        dist = manifold.dist(z1, z2)

        torch.save(z1, path / "z1")
        torch.save(z2, path / "z2")
        torch.save(dist, path / "dist")

        return cls(z1, z2, dist)


def generate_points(n, *, ndim):

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








