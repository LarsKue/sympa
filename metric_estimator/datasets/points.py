
import torch

from .file import FileDataset
from .. import io
from .. import math


class SiegelPointDataset(FileDataset):
    @classmethod
    def generate(cls, *, size, ndim, path, overwrite=False):
        path = io.prepare_path(path)
        io.handle_overwrite(path, overwrite)

        print(f"Sampling {size} pairs of Siegel points in {ndim} dimensions...")

        max_value = 0

        for i in range(size):
            z1 = generate_points(1, ndim=ndim)
            z2 = generate_points(1, ndim=ndim)

            z = math.transform_bft(z1, z2)

            mv = torch.max(z)
            if mv > max_value:
                max_value = mv

            torch.save(z1, path / ("z1_" + str(i)))
            torch.save(z2, path / ("z2_" + str(i)))
            torch.save(z, path / ("z_" + str(i)))

            print(f"\rProgress: {100.0 * (i + 1) / size:.2f}%", end="")

        print()

        instance = cls(path, size, ["z1", "z2", "z"])

        # scale to unit box
        def transform(tensors):
            z1, z2, z = tensors

            z1 = z1 / max_value
            z2 = z2 / max_value
            z = z / max_value

            return z1, z2, z

        instance.transform(how=transform)

        return instance


def generate_points(n, *, ndim):
    real = math.make_symmetric_tensor(n, ndim=ndim)
    imag = math.make_spd_standard(n, ndim=ndim)

    return torch.stack((real, imag), dim=1)
