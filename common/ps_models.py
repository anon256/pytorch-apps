import torch
import torch.nn as nn
from torch.nn import init, Parameter
from common.optimizer import PSOptimizer
from common.utils import rsetattr, rgetattr
import adapm
import math
import sys

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair


class PSDense(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self._param_buffers = {}
        self.kv = None
        self.key_offset = None
        self.opt = None
        for (name, param) in self.named_parameters():
            self._param_buffers[name] = torch.empty((2,)+param.size())

    def kv_init(self, kv: adapm.Worker, key_offset=0, opt: PSOptimizer = None, initVals=True, signalIntent=True):
        # init self
        self.kv = kv
        self.key_offset = key_offset
        self.opt = opt
        if signalIntent:
            self.intent()
        # init the densely accessed regular nn.parameter(s) first
        offset = self.key_offset

        for i, (name, param) in enumerate(self.named_parameters()):
            if initVals:
                self._param_buffers[name][0] = param.clone().detach()
                if self.opt:
                    self._param_buffers[name][1] = self.opt.initial_accumulator_value
                self.kv.set(torch.tensor([offset], dtype=torch.int64), self._param_buffers[name])
            offset += 1

        # init PS-submodules last, key_offset is already set in lens() for these layers.
        for module in self.model.modules():
            if module != self.model and hasattr(module, "kv_init"):
                module.kv_init(kv, offset, opt, initVals)
                offset += len(module.lens())

    def grad_hook(self, key: torch.Tensor, name) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = clip_grad_norm(grad, 100)
            self.opt.update_in_place(grad.cpu(), self._param_buffers[name][0], self._param_buffers[name][1])
            self.kv.push(key, self._param_buffers[name], True)
            return grad
        return hook

    def intent(self, start=0, stop=sys.maxsize):
        num_parameters = sum(1 for i in self.parameters())
        keys = torch.arange(num_parameters) + self.key_offset
        self.kv.intent(keys, start, stop)

    def pull(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([i+self.key_offset])
            self.kv.pull(key, self._param_buffers[name])
            newParam = Parameter(self._param_buffers[name][0].to(param.device, copy=True, non_blocking=True))
            newParam.register_hook(self.grad_hook(key, name))
            rsetattr(self, name, newParam)

    def forward(self, *args, **kwargs) -> (torch.Tensor, torch.BoolTensor):
        return self.model(*args, **kwargs)

    def lens(self):
        lens = torch.tensor([param.flatten().shape[0] for param in self.parameters()], dtype=torch.int64) * 2  # twice for optim params
        for module in self.model.modules():
            if module != self.model and hasattr(module, "lens"):
                lens = torch.cat((lens, module.lens()))

        return lens

    def extra_repr(self):
        return f"Dense PS-enabled:"


class PSEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        max_size: int = 2**20
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._buffer = None
        self.max_size = max_size
        self.key_offset = None

    def kv_init(self, kv: adapm.Worker, key_offset=0, opt: PSOptimizer = None, initVals=True):
        self.kv = kv
        self.key_offset = key_offset
        self.opt = opt

        if initVals:
            for ids in torch.LongTensor(range(self.num_embeddings)).split(self.max_size):
                keys = ids + self.key_offset
                values = torch.empty(keys.size()+(2, self.embedding_dim), dtype=torch.float32)
                nn.init.xavier_uniform_(PSEmbedding._embeddings(values))

                if self.opt:
                    PSEmbedding._accumulators(values)[:] = self.opt.initial_accumulator_value
                self.kv.set(keys.long(), values, True)
            self.kv.waitall()

    def _embeddings(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 0)

    def _accumulators(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 1)

    def intent(self, ids: torch.Tensor, start, stop=0):
        keys = ids.flatten() + self.key_offset
        self.kv.intent(keys, start, stop)

    def pull(self, ids: torch.Tensor):
        keys = ids.flatten() + self.key_offset
        size = ids.size() + (2, self.embedding_dim)
        self._buffer = torch.empty(size, dtype=torch.float32)
        keys = keys.cpu()
        self.kv.pull(keys, self._buffer)

    def forward(self, ids: torch.Tensor, device=None):
        if self._buffer is None:
            self.pull(ids)

        embeddings = PSEmbedding._embeddings(self._buffer).to(device=device).requires_grad_()
        if self.training and self.opt:
            embeddings.register_hook(self.grad_hook(ids))
        elif not self.training:
            self._buffer = None

        return embeddings

    def grad_hook(self, ids: torch.Tensor) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = clip_grad_norm(grad, 100)
            keys = ids.flatten() + self.key_offset
            self.opt.update_in_place(grad.cpu(), PSEmbedding._embeddings(self._buffer), PSEmbedding._accumulators(self._buffer))
            self.kv.push(keys.cpu(), self._buffer.cpu(), True)
            self._buffer = None

            return grad
        return hook

    def lens(self):
        return torch.ones(self.num_embeddings, dtype=torch.int64) * self.embedding_dim * 2  # twice embedding_dim for optim params

    def extra_repr(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class PSEmbedGraphConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm='both',
        bias=True,
        activation=None,
        allow_zero_in_degree=False
    ) -> None:
        super(PSEmbedGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._activation = activation
        self.embedding = PSEmbedding(
            num_embeddings=in_feats,
            embedding_dim=out_feats,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.register_parameter('bias', None)

    def intent(self, ids: torch.Tensor, start, stop = 0):
        self.embedding.intent(ids, start, stop)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)

            feat_src = self.embedding(feat_src.long(), self.bias.device)

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}, normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

def clip_grad_norm(
        grad: torch.Tensor, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of a tensor, based on torch.clip_grad_norm_. Works slightly different than the original clip_grad_norm_: does not work in place, returns the clipped gradient instead, and does not support all norms (e.g., 0-norm is not supported) """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grad.device
    if norm_type == torch.inf:
        total_norm = grad.detach().abs().max().to(device)
    else:
        total_norm = torch.norm(grad.detach(), norm_type).to(device)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad.detach().mul(clip_coef_clamped.to(grad.device))

