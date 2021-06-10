
from torch import nn


class ModelMixin:
    """
    Neural Net
    """
    def __init__(self, *layers):
        self.model = nn.Sequential(*layers)
