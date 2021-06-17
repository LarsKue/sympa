
from torch import nn
from .device import device


class ModelMixin(nn.Module):
    """
    Neural Net
    """
    def __init__(self, model):
        super(ModelMixin, self).__init__()
        self.model = model
        self.model.to(device)

    @classmethod
    def from_layers(cls, *layers):
        model = nn.Sequential(*layers)
        cls(model)
