import math
import torch as th
from torch import Tensor


class ChanLinear(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = th.nn.Parameter(
            th.empty((out_channels, in_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = th.nn.Parameter(th.empty(out_channels, in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        th.nn.init.uniform_(
            self.weight,
            1/math.sqrt(self.out_channels * self.in_features),
            1/math.sqrt(self.out_channels * self.in_features)
        )
        if self.bias is not None:
            th.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        y = th.einsum("nij,...mj->...nmi", self.weight, x).sum(dim=-2)

        if self.bias is not None:
            return y + self.bias

        return y
    
    
class RegChanLinear(ChanLinear):
    def __init__(self, in_channels: int, out_channels: int, in_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_channels, out_channels, in_features, bias, device, dtype)
        self.register_buffer(
            "mask",
            th.empty(in_features, in_features, **factory_kwargs)
        )
        self.mask.fill_(1)
        self.mask.triu_()
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x_masked = (x.diag_embed() @ self.mask).mT
        y = th.einsum("nij,...mji->...nmi", self.weight, x_masked.mT).sum(dim=-2)
        
        if self.bias is not None:
            return y + self.bias
        
        return y


class CondGateAutoRegLinear(th.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reg_lin1 = RegChanLinear(*args, **kwargs)
        self.reg_lin2 = RegChanLinear(*args, **kwargs)
        self.lin1 = ChanLinear(*args, **kwargs)
        self.lin2 = ChanLinear(*args, **kwargs)
        self.tanh = th.nn.Tanh()
        self.sigmoid = th.nn.Sigmoid()
    
    def forward(self, x, h):
        inp = self.tanh(self.reg_lin1(x) + self.lin1(h))
        gate = self.tanh(self.reg_lin2(x) + self.lin2(h))
        return inp * gate


class PixelLinear(th.nn.Module):
    def __init__(
            self, num_embedding, in_feature, in_channels=1, out_channels=10,
            n_layers=5, bias=True, device="cpu"
    ):
        super().__init__()
        self.layers = th.nn.ModuleList()
        # input layer
        self.layers.append(
            CondGateAutoRegLinear(in_channels, out_channels, in_feature, bias)
        )
        # hidden layers
        for i in range(1, n_layers+1):
            self.layers.append(
                CondGateAutoRegLinear(out_channels, out_channels, in_feature, bias)
            )
            self.layers.append(
                th.nn.BatchNorm1d(out_channels)
            )
            if i == n_layers:
                self.layers.append(th.nn.Tanh())
            else:
                self.layers.append(th.nn.LeakyReLU())
            
        # output layer
        self.layers.append(
            CondGateAutoRegLinear(out_channels, num_embedding, in_feature, bias)
        )
        self.to(device=device)
    
    def forward(self, x, h):
        out = x
        for layer in self.layers:
            if isinstance(layer, CondGateAutoRegLinear):
                out = layer(out, h)
            else:
                out = layer(out)
        return out
    
            