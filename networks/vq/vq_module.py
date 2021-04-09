# the original script came courtesy of moskomule (https://github.com/moskomule).
import os
import numpy as np
import importlib.util

from typing import Optional
from typing import Tuple
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .grad_approximation import custom_straight_through_estimator


torch.set_printoptions(threshold=np.inf)


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


if is_distributed():
    from torch.distributed import all_reduce


def is_faiss_available() -> bool:
    _faiss_available = importlib.util.find_spec("faiss") is not None
    if _faiss_available:
        import faiss
        if not hasattr(faiss, 'StandardGpuResources'):
            print("faiss is available but is not for GPUs")
    return _faiss_available


def _tensor_to_ptr(input: torch.Tensor
                   ) -> Any:
    import faiss

    assert input.is_contiguous()
    assert input.dtype in [torch.float32, torch.int64]
    if input.dtype is torch.float32:
        return faiss.cast_integer_to_float_ptr(input.storage().data_ptr() + input.storage_offset() * 4)
    else:
        return faiss.cast_integer_to_long_ptr(input.storage().data_ptr() + input.storage_offset() * 8)


def _torch_knn(keys: torch.Tensor,
               queries: torch.Tensor,
               num_neighbors: int,
               distance: str
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if distance == "dot_product":
            scores = keys.mm(queries.t())
        else:
            scores = keys.mm(queries.t())
            scores *= 2
            scores -= (keys.pow(2)).sum(1, keepdim=True)
            scores -= (queries.pow(2)).sum(1).unsqueeze_(0)
        scores, indices = scores.topk(k=num_neighbors, dim=0, largest=True)
        scores = scores.t()
        indices = indices.t()

    return scores, indices


def _faiss_knn(keys: torch.Tensor,
               queries: torch.Tensor,
               num_neighbors: int,
               distance: str
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    # https://github.com/facebookresearch/XLM/blob/master/src/model/memory/utils.py
    if not is_faiss_available():
        raise RuntimeError("faiss_knn requires faiss-gpu")
    import faiss

    metric = faiss.METRIC_INNER_PRODUCT if distance == 'dot_product' else faiss.METRIC_L2

    k_ptr = _tensor_to_ptr(keys)
    q_ptr = _tensor_to_ptr(queries)

    scores = keys.new_zeros((queries.size(0), num_neighbors), dtype=torch.float32)
    indices = keys.new_zeros((queries.size(0), num_neighbors), dtype=torch.int64)

    s_ptr = _tensor_to_ptr(scores)
    i_ptr = _tensor_to_ptr(indices)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = num_neighbors
    args.dims = queries.size(1)
    args.vectors = k_ptr
    args.vectorsRowMajor = True
    args.numVectors = keys.size(0)
    args.queries = q_ptr
    args.queriesRowMajor = True
    args.numQueries = queries.size(0)
    args.outDistances = s_ptr
    args.outIndices = i_ptr
    faiss.bfKnn(FAISS_RES, args)
    return scores, indices


def k_nearest_neighbor(keys: torch.Tensor,
                       queries: torch.Tensor,
                       num_neighbors: int,
                       distance: str, *,
                       backend: Optional[str] = "torch"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ k-Nearest Neighbor search
    :param keys: tensor of (num_keys, dim)
    :param queries: tensor of (num_queries, dim)
    :param num_neighbors: `k`
    :param distance: name of distance (`dot_product` or `l2`)
    :param backend: backend (`faiss` or `torch`)
    :return: scores, indices
    """

    assert backend in {"faiss", "torch"}
    assert distance in {'dot_product', 'l2'}
    assert keys.size(1) == queries.size(1)
    f = _faiss_knn if backend == "faiss" and is_faiss_available() else _torch_knn
    return f(keys, queries, num_neighbors, distance)


if is_faiss_available():
    import faiss

    FAISS_RES = faiss.StandardGpuResources()
    FAISS_RES.setDefaultNullStreamAllDevices()
    FAISS_RES.setTempMemory(1200 * 1024 * 1024)


def exponential_moving_average_(base: torch.Tensor,
                                update: torch.Tensor,
                                momentum: float
                                ) -> torch.Tensor:
    return base.mul_(momentum).add_(update, alpha=1 - momentum)


class VQModule(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 dict_size: int,
                 momentum: float,
                 eps: float,
                 knn_backend: Optional[str]
                 ) -> None:
        super(VQModule, self).__init__()
        self.emb_dim = emb_dim
        self.dict_size = dict_size
        self.momentum = momentum
        self.eps = eps
        self._knn_backend = knn_backend

        embed = torch.randn(self.dict_size, self.emb_dim)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(self.dict_size))
        self.register_buffer('embed_avg', self.embed.T.clone())

    def forward(self,
                input: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        quantized, ids = self._quantize(input)
        commit_loss = F.mse_loss(input, quantized)
        quantized = custom_straight_through_estimator(quantized, input)
        return quantized, commit_loss, ids

    @torch.no_grad()
    def _quantize(self,
                  input: torch.Tensor):
        # flatten (BHW)xC
        flatten = input.transpose(1, -1).reshape(-1, self.emb_dim)
        # dist: (BHW)x1, ids: (BHW)x1
        dist, ids = k_nearest_neighbor(self.embed, flatten, 1, 'l2', backend=self._knn_backend)
        # embed_onthot: (BHW)x{dict_size}
        embed_onehot = F.one_hot(ids.view(-1), self.dict_size).to(flatten.dtype)
        # quantized: -> BxCxHxW, ids: -> BxHxW
        b, c, h, w = input.size()
        ids = ids.view(b, h, w)
        quantized = self.lookup(ids).transpose(1, -1)

        if self.training:
            # embed_onehot_sum: {dict_size}
            embed_onehot_sum = embed_onehot.sum(dim=0)
            # embed_sum: Cx{dict_size}
            embed_sum = flatten.T @ embed_onehot

            if is_distributed():
                all_reduce(embed_onehot)
                all_reduce(embed_sum)
                ws = get_world_size()
                embed_onehot /= ws
                embed_sum /= ws

            exponential_moving_average_(self.cluster_size, embed_onehot_sum, self.momentum)
            exponential_moving_average_(self.embed_avg, embed_sum, self.momentum)

            n = self.cluster_size.sum()
            cluster_size = n * (self.cluster_size + self.eps) / (n + self.dict_size * self.eps)
            self.embed.copy_(self.embed_avg.T / cluster_size.unsqueeze(1))

        return quantized, ids

    def lookup(self,
               ids: torch.Tensor
               ) -> torch.Tensor:
        return F.embedding(ids, self.embed)
