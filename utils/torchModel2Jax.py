import sys
import torch as th
import numpy as np
import jax.numpy as jnp
from torch import nn, Tensor
from collections import defaultdict
from model.DOptimizer import InitializerModule


class LinearJax:
    def __init__(self, weight, bias):
        self.weight, self.bias = weight.detach().numpy(), bias.detach().numpy()

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class BatchNorm1dJax:
    def __init__(self, mean, var, gamma, beta, eps=1e-05):
        self.eps = eps
        self.mean, self.var, self.gamma, self.beta = mean.detach().numpy(), var.detach().numpy(), gamma.detach().numpy(), beta.detach().numpy()

    def __call__(self, x):
        return ((x - self.mean) / jnp.sqrt(self.var + self.eps) * self.gamma) + self.beta

class ReLUJax:
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return jnp.maximum(0, x) 

def extract_weights(model):
    WandB = defaultdict(dict)
    state = model.state_dict()

    for key in list(state.keys()):
        WandB[int(key.split(".")[0])][key.split(".")[-1]] = state[key]

    return WandB


class SequentialJax:
    def __init__(self, model):
        WandB = extract_weights(model)
        self.net = [
            LinearJax(WandB[0]["weight"], WandB[0]["bias"]),
            BatchNorm1dJax(WandB[1]["running_mean"], WandB[1]["running_var"], WandB[1]["weight"], WandB[1]["bias"]),
            ReLUJax()
        ]
    
    def __call__(self, x):
        y = x
        for layer in self.net:
            y = layer(y)
        return y
    
    
class InitializerModuleJax:
    def __init__(self, mean, std, mlp_shape, model):
        self.mean = mean
        self.std = std
        
        out_layer = model.net[-1]
        self.net = [
            *[
                SequentialJax(model.net[i])
                for i in range(1 + len(mlp_shape))
            ],
            LinearJax(out_layer.weight, out_layer.bias),
        ]

    def __call__(self, primal_sol_1, observations):
        observations = (observations - self.mean) / self.std
        y = jnp.hstack([primal_sol_1, observations])
        
        for layer in self.net:
            y = layer(y)

        return y


class VQDecoderJax:
    def __init__(self):
        pass
    
    def __call__(self):
        pass


if __name__ == "__main__":
    torch_model = InitializerModule(1, 1, 77, 44, [256, 1024, 1024, 1024])
    jax_model = InitializerModuleJax(1, 1, [256, 1024, 1024, 1024], torch_model)
    print(jax_model(jnp.ones((1, 22)), jnp.ones((1, 55))).shape)