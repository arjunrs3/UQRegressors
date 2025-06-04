import torch.nn as nn


def get_activation(name: str):
    name = name.lower()
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "none": nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]
