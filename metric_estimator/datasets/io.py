
import pathlib as pl
import shutil


def prepare_path(path):
    if not isinstance(path, pl.Path):
        path = pl.Path(path)

    return path


def handle_overwrite(path, overwrite=False):
    if not overwrite and path.exists():
        raise FileExistsError(f"Cannot overwrite {path}.")
    if overwrite and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)
