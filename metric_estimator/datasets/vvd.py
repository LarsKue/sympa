
import torch

from .points import SiegelPointDataset
from .file import FileDataset
from .io import prepare_path


class VVDDataset(FileDataset):
    @classmethod
    def generate(cls, *, size, manifold, path, overwrite=False):
        path = prepare_path(path, overwrite)
        points = SiegelPointDataset.generate(size=size, ndim=manifold.ndim, path=path, overwrite=overwrite)

        print(f"Calculating {size} distances...")
        for i, (z1, z2) in enumerate(points):
            vvd = manifold.vvd(z1, z2)

            torch.save(vvd, path / ("vvd_" + str(i)))

            print(f"\rProgress: {100.0 * (i + 1) / size}%", end="")

        print()

        return cls(path=path, size=size, attribute_names=points.attribute_names + ["vvd"])

