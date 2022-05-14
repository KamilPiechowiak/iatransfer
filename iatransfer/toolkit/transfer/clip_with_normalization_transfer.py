import torch
from torch import nn

from iatransfer.toolkit.transfer.clip_transfer import ClipTransfer
from iatransfer.toolkit.transfer.transfer_stats import TransferStats


class ClipWithNormalizationTransfer(ClipTransfer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def postprocess(self, layer: nn.Module) -> TransferStats:
        if layer.weight is None:
            return
        if any([isinstance(layer, clazz) for clazz in [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]]):
            new_var = 1.0 * layer.groups
            for dimension in layer.weight.shape:
                new_var /= dimension
            new_var *= layer.weight.shape[1]
            new_var *= 2  # relu gain
        elif isinstance(layer, nn.Linear):
            new_var = 1 / (layer.weight.shape[0] + layer.weight.shape[1])
            new_var /= 3  # to have the same variance as Unif(-sqrt(k), sqrt(k)) see: pytorch docs
        else:
            return

        layer.weight *= new_var**0.5 / torch.std(layer.weight)
        if layer.bias is not None:
            layer.bias *= new_var**0.5 / torch.std(layer.bias)
        print("normalized")
