import torch
import numpy as np

from .device import device


def is_positive_definite(z):
    """
    One of the fastest ways to check for positive definiteness
    is to try a cholesky decomposition, which fails if the matrix
    is not positive definite
    works for matrices of shape (b, n, n) or (n, n)
    where b is the batch size and n is the dimension
    """
    try:
        torch.linalg.cholesky(z)
        return True
    except RuntimeError:
        return False


def is_symmetric(z):
    """
    Check if matrices of shape (b, n, n) or (n, n) are symmetric
    where b is the batch size and n is the dimension
    """
    return torch.allclose(z.transpose(-2, -1), z)


def symmetricize(x, copy=True):
    if copy:
        # symmetricize by copying the upper
        # triangular part into the lower one
        return x.triu() + x.triu(1).transpose(-1, -2)
    # symmetricize by matrix multiplication with transposed self
    return torch.matmul(x.transpose(-2, -1), x)


def bft(x):
    """
    Batched and Flattened Upper Triangular Part of Complex Matrices

    Takes a batch of complex (symmetric) matrices
    x of shape (batch_size, 2, ndim, ndim)
    and returns the flattened upper triangular part
    of each sample in the shape (batch_size, ndim * (ndim + 1))
    """
    batch_size = x.shape[0]
    ndim = x.shape[-1]

    input_shape = (batch_size, 2, ndim, ndim)
    output_shape = (batch_size, ndim * (ndim + 1))

    if x.shape != input_shape:
        raise ValueError(f"Expected Input Shape {input_shape}, but got {x.shape}.")

    rows, columns = torch.triu_indices(ndim, ndim)

    return x[:, :, rows, columns].reshape(output_shape)


def ibft(x):
    """
    Inverse to :func:`~bft`
    """
    batch_size = x.shape[0]
    s = np.sqrt(4 * x.shape[1] + 1)
    ndim = int(0.5 * (-1 + s))

    input_shape = (batch_size, ndim * (ndim + 1))

    if x.shape != input_shape:
        raise ValueError(f"Expected Input Shape {input_shape}, but got {x.shape}")

    nu = int(ndim * (ndim + 1) / 2)
    real_u, imag_u = x.split(nu, dim=1)

    shape = (batch_size, ndim, ndim)

    real = torch.zeros(shape, device=x.device)
    imag = torch.zeros(shape, device=x.device)

    rows, columns = torch.triu_indices(ndim, ndim)

    real[:, rows, columns] = real_u
    imag[:, rows, columns] = imag_u

    real = symmetricize(real, copy=True)
    imag = symmetricize(imag, copy=True)

    return torch.stack((real, imag), dim=1)


def transform_bft(z1, z2):
    zz1 = bft(z1)
    zz2 = bft(z2)

    z = torch.cat((zz1, zz2), dim=-1).squeeze()

    return z


def transform_ibft(z):
    n = z.shape[-1] // 2
    zz1, zz2 = z.split(n, dim=-1)

    z1 = ibft(zz1)
    z2 = ibft(zz2)

    return z1, z2


def mare(yhat, y):
    """
    Mean Absolute Relative Error
    @param yhat: predicted value
    @param y: ground truth
    """
    return torch.mean(torch.abs(yhat - y) / torch.abs(y), dim=0)


def make_symmetric_tensor(n, *, ndim):
    shape = (n, ndim, ndim)
    t = torch.rand(shape, device=device)

    return symmetricize(t, copy=False)


def make_spd_skewed(n, *, ndim):
    t = make_symmetric_tensor(n, ndim=ndim)

    evs = torch.symeig(t)
    e = -evs.eigenvalues[:, 0] + 1
    e = e.unsqueeze(-1).unsqueeze(-1)

    diag = e * torch.eye(ndim, device=device)
    t.add_(diag)

    return t


def make_spd_tensor(n, *, ndim):
    """
    Generate batched random Symmetric Positive Definite (SPD) matrices
    Inspired by `sklearn.datasets.make_spd_matrix`
    Significantly faster than comparative algorithms
    :return: SPD tensor of shape (n, ndim, ndim)
    """
    # generate random symmetric matrix
    t = make_symmetric_tensor(n, ndim=ndim)

    # singular value decomposition
    u, _, vh = torch.linalg.svd(t)

    s = 1.0 + torch.diag_embed(torch.rand(n, ndim, device=device))

    # reconstruct with all positive eigenvalues
    t = torch.matmul(torch.matmul(u, s), vh)

    return t


def make_spd_standard(n, *, ndim):
    """
    Generate batched random Symmetric Positive Definite (SPD) matrices
    as described by Prof. Köthe as the standard algorithm
    :return: SPD tensor of shape (n, ndim, ndim)
    """
    shape = (n, ndim, ndim)

    # sample random matrix
    g = torch.randn(shape, device=device)

    # compute QR decomposition
    q, r = torch.linalg.qr(g)
    qt = torch.transpose(q, -2, -1)

    # sample positive eigenvalues from uniform distribution
    eigvals = torch.abs(torch.rand(n, ndim, device=device))

    # embed eigenvalues into diagonal matrices L
    l = torch.diag_embed(eigvals)

    # reconstruct from eigen-decomposition S = U * L * U^T
    # where U = Q
    s = torch.matmul(torch.matmul(q, l), qt)

    return s
