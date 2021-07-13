import warnings

import torch


class Loss:
    """
    Base Loss Class
    Expects Input like (batch_size, feature_size) or (batch_size,)
    Returns Reduced Output like ()
    Where typically the value is the mean loss over the batch
    """
    def __init__(self, reduction=torch.mean):
        self.reduction = reduction

    def __call__(self, predicted, true, dim=0):
        if predicted.shape != true.shape:
            warnings.warn(f"Loss Input has incompatible shape: {predicted.shape} and {true.shape}.")
        predicted, true = torch.broadcast_tensors(predicted, true)
        losses = self.forward(predicted, true)
        return self.reduce(losses, dim=dim)

    def forward(self, predicted: torch.Tensor, true: torch.Tensor):
        raise NotImplementedError

    def reduce(self, losses, dim=0):
        return self.reduction(losses, dim=dim)


class MARELoss(Loss):
    """
    Mean Absolute Relative Error
    """
    def forward(self, predicted, true):
        return torch.abs(predicted - true) / torch.abs(true)


class L2Loss(Loss):
    def forward(self, predicted, true):
        if predicted.dim() == 1:
            return torch.abs(predicted - true)
        return torch.sqrt(torch.sum(torch.square(predicted - true), dim=1))
