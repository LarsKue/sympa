
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pathlib as pl

from .device import device


class MetricDistanceSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        meta = pd.read_csv(csv_file)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class MetricDistanceLoader(DataLoader):
    pass


def symmetricize(x):
    # make a tensor symmetric by copying the
    # upper triangular part into the lower one
    return x.triu() + x.triu(1).transpose(-1, -2)


def generate_points(n, ndim):
    shape = (n, ndim, ndim)

    real = torch.randn(shape, device=device)
    imag = torch.randn(shape, device=device)

    # symmetricizing preserves the distribution
    real = symmetricize(real)
    imag = symmetricize(imag)

    # make imaginary part positive definite
    # by adding smallest eigenvalue on the diagonal
    evs = torch.symeig(imag)
    e = evs.eigenvalues[:, 0].unsqueeze(-1).unsqueeze(-1) + 1e-6
    diag = e * torch.eye(ndim, device=device)
    print(diag.shape, imag.shape, shape)

    imag.add_(diag)

    return torch.stack((real, imag), dim=1)


def generate(n, *, manifold, path):
    if path.is_dir():
        raise ValueError("Please choose a filename, not a directory.")
    if path.exists():
        raise FileExistsError(f"Cannot overwrite file {path}")

    z1 = generate_points(n, manifold.ndim)
    z2 = generate_points(n, manifold.ndim)
    dists = manifold.dist(z1, z2)

    # TODO: pickle

    print(dists)

