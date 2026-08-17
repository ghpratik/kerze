"""
kerze.nn — public API surface for the neural-network layer.

Usage:
    from kerze import nn
    model = nn.Sequential(nn.Linear(3, 8), nn.Sigmoid(), nn.Linear(8, 1))
"""

from kerze.nn.module import Module
from kerze.nn.parameter import Parameter
from kerze.nn.linear import Linear
from kerze.nn.activation import ReLU, GELU, Sigmoid, Tanh
from kerze.nn.loss import MSELoss
from kerze.nn.container import Sequential
from kerze.nn import functional
from kerze.nn import init

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "MSELoss",
    "Sequential",
    "functional",
    "init",
]