
import torch

from .file import FileDataset
from .io import prepare_path, handle_overwrite
from .. import math


class SiegelPointDataset(FileDataset):
    @classmethod
    def generate(cls, *, size, ndim, path, overwrite=False):
        path = prepare_path(path)
        handle_overwrite(path, overwrite)

        print(f"Sampling {size} pairs of Siegel points in {ndim} dimensions...")

        for i in range(size):
            z1 = generate_points(1, ndim=ndim)
            z2 = generate_points(1, ndim=ndim)

            z = math.transform_bft(z1, z2)

            torch.save(z1, path / ("z1_" + str(i)))
            torch.save(z2, path / ("z2_" + str(i)))
            torch.save(z, path / ("z_" + str(i)))

            print(f"\rProgress: {100.0 * (i + 1) / size:.2f}%", end="")

        print()
        return cls(path, size, ["z1", "z2", "z"])


def generate_points(n, *, ndim):
    real = math.make_symmetric_tensor(n, ndim=ndim)
    imag = math.make_spd_standard(n, ndim=ndim)

    z = torch.stack((real, imag), dim=1)

    return math.scale_to_unit_box(z)
