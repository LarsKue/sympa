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

    def __call__(self, predicted, true):
        if predicted.shape != true.shape:
            warnings.warn(f"Loss Input has incompatible shape: {predicted.shape} and {true.shape}.")
        if predicted.dim() == 1 or true.dim() == 1:
            warnings.warn(f"Loss Input has possibly incompatible dimensions: {predicted.shape} and {true.shape}.")
        predicted, true = torch.broadcast_tensors(predicted, true)
        losses = self.forward(predicted, true)
        return self.reduce(losses)

    def forward(self, predicted: torch.Tensor, true: torch.Tensor):
        raise NotImplementedError

    def reduce(self, losses):
        return self.reduction(losses)


class MARE(Loss):
    """
    Mean Absolute Relative Error
    """
    def forward(self, predicted, true):
        return torch.abs(predicted - true) / torch.abs(true)


class L2Loss(Loss):
    def forward(self, predicted, true):
        return torch.linalg.vector_norm(predicted - true, dim=-1)


class LogCoshLoss(Loss):
    def forward(self, predicted, true):
        return torch.log(torch.cosh(predicted - true))


class MSLE(Loss):
    """
    Mean Squared Log-Scaled Error
    """

    def __init__(self, alpha=0.01, epsilon=1e-9, reduction=torch.mean):
        super(MSLE, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, predicted, true):
        mse = torch.square(predicted - true)
        msle = torch.square(torch.log(predicted + self.epsilon) - torch.log(true + self.epsilon))

        return mse + self.alpha * msle
