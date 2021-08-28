
import pickle

from metric_estimator.legacy import io


class LossHistory:
    def __init__(self, train_loss=None, val_loss=None, complete=True):
        if train_loss is None:
            train_loss = []
        if val_loss is None:
            val_loss = []

        self.train_loss = train_loss
        self.val_loss = val_loss
        self.complete = complete

        self.__validate()

    def step(self, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def save(self, path, overwrite=False):
        path = io.prepare_path(path)
        io.handle_overwrite(path, overwrite)
        with open(path / "train_loss", "wb+") as f:
            pickle.dump(self.train_loss, f)

        with open(path / "val_loss", "wb+") as f:
            pickle.dump(self.val_loss, f)

    def load(self, path):
        path = io.prepare_path(path)
        with open(path / "train_loss", "rb") as f:
            self.train_loss = pickle.load(f)
        with open(path / "val_loss", "rb") as f:
            self.val_loss = pickle.load(f)

    def __validate(self):
        """ Ensure train and validation history are of same length, if complete """
        if not self.complete:
            return

        nt = len(self.train_loss)
        nv = len(self.val_loss)
        if nt != nv:
            raise ValueError(f"Unequal number of history entries in training vs. validation: {nt} vs {nv}.")

    def __len__(self):
        nt = len(self.train_loss)
        nv = len(self.val_loss)

        return max(nt, nv)