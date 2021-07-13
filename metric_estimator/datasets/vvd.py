
import torch

from .points import SiegelPointDataset
from .file import FileDataset
from .. import io


class VVDDataset(FileDataset):
    @classmethod
    def generate(cls, *, size, manifold, path, overwrite=False):
        path = io.prepare_path(path)
        points = SiegelPointDataset.generate(size=size, ndim=manifold.ndim, path=path, overwrite=overwrite)

        print(f"Calculating {size} distances...")
        for i, (z1, z2, _) in enumerate(points):
            vvd = manifold.vvd(z1, z2)
            vvd = torch.squeeze(vvd)

            torch.save(vvd, path / ("vvd_" + str(i)))

            print(f"\rProgress: {100.0 * (i + 1) / size:.2f}%", end="")

        print()

        return cls(path=path, size=size, attribute_names=points.attribute_names + ["vvd"])

