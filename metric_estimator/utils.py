
import torch


def symmetricize(x):
    # make a tensor symmetric by copying the
    # upper triangular part into the lower one
    return x.triu() + x.triu(1).transpose(-1, -2)


def flat_batched_complex_triu(x):
    """
    Takes a batch of complex (symmetric) matrices
    x of shape (batch_size, 2, ndim, ndim)
    and returns the flattened upper triangular part
    of each sample in the shape (batch_size, ndim * (ndim + 1))
    """
    batch_size = x.shape[0]
    ndim = x.shape[-1]

    input_shape = (batch_size, 2, ndim, ndim)
    output_shape = (batch_size, -1)

    if x.shape != input_shape:
        raise ValueError(f"Expected Input Shape {input_shape}, but got {x.shape}.")

    rows, columns = torch.triu_indices(ndim, ndim)

    return x[:, :, rows, columns].reshape(output_shape)

