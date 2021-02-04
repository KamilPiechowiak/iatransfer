from typing import List, Union

import torch.nn as nn

from iatransfer.toolkit.base_standardization import Standardization


class FlattenStandardization(Standardization):
    """Basic flatten standardization algorithm for IAT.
    """

    def standardize(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        classes = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.Linear
        ]
        if any([isinstance(module, clazz) for clazz in classes]):
            return [module]

        layers = []
        for child in module.children():
            layers += self.standardize(child)
        if len(layers) == 0:
            layers = [module]
        return layers
