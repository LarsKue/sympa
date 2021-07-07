
import pathlib as pl
import shutil


def prepare_path(path, overwrite=False):
    if not isinstance(path, pl.Path):
        path = pl.Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"Cannot overwrite {path}.")
    if overwrite and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)

    return path
