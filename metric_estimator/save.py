
import torch
import pathlib as pl
import shutil

from .model import ModelMixin


class ModelSaveMixin(ModelMixin):
    def save_model(self, path, overwrite=False):
        if not isinstance(path, pl.Path):
            path = pl.Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Cannot overwrite {path}.")
        if overwrite:
            path.unlink(missing_ok=True)

        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load_model(path):
        model = torch.nn.Module()
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

        return model
