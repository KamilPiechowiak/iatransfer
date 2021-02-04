from typing import List

from torch import nn


def flatten_modules(module: nn.Module) -> List[nn.Module]:
    classes = [
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Linear
    ]
    if any([isinstance(module, clazz) for clazz in classes]):
        return [module]

    layers = []
    for child in module.children():
        layers.extend(flatten_modules(child))
    if len(layers) == 0:
        layers.append(module)
    return layers
