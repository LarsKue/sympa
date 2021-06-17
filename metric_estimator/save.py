
import torch
import pathlib as pl
import shutil

from .model import ModelMixin

# TODO: test
class ModelSaveMixin(ModelMixin):
    def save(self, path, overwrite=False):
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Cannot overwrite {path}.")

        if overwrite:
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path):
        model = torch.nn.Module()
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

        return cls(model)
