import torch
import torch.nn as nn
from sympa.utils import get_logging
from sympa import config
import geoopt as gt
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from sympa.math import symmetric_math as smath
import abc

log = get_logging()


class Embeddings(nn.Module, abc.ABC):
    """Abstract Embedding layer that operates with embeddings as a Manifold parameter"""

    def __init__(self, num_embeddings, embedding_dim, manifold, _embeds):
        """
        :param num_embeddings: number of elements in the table
        :param embedding_dim: dimensionality of embeddings.
        If it is vector embeddings, it is the amount of dimensions in the vector.
        If it is Matrix embeddings, it is the last dimension on the matrix
        :param manifold: Instance of Geoopt Manifold
        :param _embeds: tensor with initial value for embeddings
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        self.embeds = gt.ManifoldParameter(_embeds, manifold=self.manifold)

    def forward(self, input_index):
        """
        :param input_index: tensor of b x 1
        :return: Embeddings of b x * x n
        """
        return self.embeds[input_index]

    def proj_embeds(self):
        """Projects embeddings back into the manifold"""
        with torch.no_grad():
            self.embeds.data = self.manifold.projx(self.embeds.data)

    def check_all_points(self):
        for i in range(len(self.embeds)):
            point = self.embeds.data[i]
            ok, reason = self.manifold.check_point_on_manifold(point, explain=True)
            if not ok:
                return False, point, reason
        return True, None, None

    @abc.abstractmethod
    def norm(self):
        pass


class ComplexSymmetricMatrixEmbeddings(Embeddings):

    def __init__(self, num_embeddings, embedding_dim, manifold):
        """
        Represents Embeddings of Complex Symemtric Matrices.
        The shape of the embedding table is: num_embeddings x 2 x embedding_dim x embedding_dim
        The second dimension is 2, since the element 0 represents the real part,
        and the element 1 represents the imaginary part of the matrices.

        :param num_embeddings: number of elements in the table
        :param embedding_dim: dimensionality of matrix embeddings.
        :param manifold: UpperHalfManifold or BoundedDomainManifold
        """
        _embeds = manifold.random(num_embeddings, from_=-config.INIT_EPS, to=config.INIT_EPS)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)

    def norm(self):
        points = self.embeds.data
        points = points.reshape(len(points), -1)
        return points.norm(dim=-1)


class VectorEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold):
        init_eps = config.INIT_EPS
        _embeds = torch.Tensor(num_embeddings, embedding_dim).uniform_(-init_eps, init_eps)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)
        self.proj_embeds()

    def norm(self):
        return self.embeds.data.norm(dim=-1)


class Model(nn.Module):
    """Graph embedding model that operates on different manifolds"""

    def __init__(self, id2node, args):
        """
        :param id2node: dict of {node_index: node_name}
        :param args: extra arguments
        """
        super().__init__()

        self.id2node = id2node
        num_points = len(id2node)
        if args.model == "euclidean":
            self.embeddings = VectorEmbeddings(num_points, args.dims, manifold=gt.Euclidean(1))
        elif args.model == "poincare":
            self.embeddings = VectorEmbeddings(num_points, args.dims, manifold=gt.PoincareBall())
        elif args.model == "upper":
            self.embeddings = ComplexSymmetricMatrixEmbeddings(num_points, args.dims,
                                                               manifold=UpperHalfManifold(args.dims))
        elif args.model == "bounded":
            self.embeddings = ComplexSymmetricMatrixEmbeddings(num_points, args.dims,
                                                               manifold=BoundedDomainManifold(args.dims))
        self.manifold = self.embeddings.manifold

    def forward(self, input_triplet):
        """
        Calculates and returns the distance in the manifold of the points given as a
        triplet of the form: (src_id, dst_id, graph_distance)
        'graph_distance' is ignored in this case.

        :param input_triplet: tensor with indexes of embeddings to process. (b, 3)
        :return: distances b
        """
        src_index, dst_index = input_triplet[:, 0], input_triplet[:, 1]
        src_embeds = self.embeddings(src_index)                       # b x 2 x n x n or b x n
        dst_embeds = self.embeddings(dst_index)                       # b x 2 x n x n

        distances = self.distance(src_embeds, dst_embeds)
        return distances

    def distance(self, src_embeds, dst_embeds):
        """
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In complex matrix spaces, it will be of the shape b x 2 x n x n. In Vector spaces it will be b x n
        :return: tensor of b with distances from each src to each dst
        """
        return self.manifold.dist(src_embeds, dst_embeds)   # b x 1

    def check_all_points(self):
        return self.embeddings.check_all_points()

    def embeds_norm(self):
        return self.embeddings.norm()
