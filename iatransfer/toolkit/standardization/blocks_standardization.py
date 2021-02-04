from typing import List, Tuple, Union

import torch.nn as nn

from iatransfer.toolkit.base_standardization import Standardization


class BlocksStandardization(Standardization):
    """Block standardization algorithm for IAT.
    """

    def standardize(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        return self.flatten_with_blocks(module, 0)[2]

    def flatten_with_blocks(self, module: nn.Module, level: int = 0) -> Tuple[int, int, List[nn.Module]]:
        depth, dimensional_layers_num, layers = 0, 0, []
        classes = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.Linear
        ]
        dimensional_layers = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.Linear
        ]
        if any([isinstance(module, clazz) for clazz in classes]):
            layers = module
            if any([isinstance(module, clazz) for clazz in dimensional_layers]):
                dimensional_layers_num += 1
        else:
            blocks = 0
            single_layers = 0
            children = list(module.children())
            if len(children) == 1:
                next_level = level
            else:
                next_level = level + 1
            for child in module.children():
                child_depth, child_dimensional_layers_num, child_layers = self.flatten_with_blocks(child, next_level)
                dimensional_layers_num += child_dimensional_layers_num
                if child_depth > 1 or (isinstance(child_layers, list) and child_dimensional_layers_num < 2):
                    for child_layer in child_layers:
                        if isinstance(child_layer, list):
                            blocks += 1
                        else:
                            single_layers += 1
                        layers += [child_layer]
                else:
                    if isinstance(child_layers, list):
                        blocks += 1
                    else:
                        single_layers += 1
                    layers += [child_layers]

            if single_layers > blocks and level > 0:
                new_layers = []
                for child_layers in layers:
                    if isinstance(child_layers, list):
                        new_layers += child_layers
                    else:
                        new_layers += [child_layers]
                layers = new_layers
                depth = 1
            else:
                depth = 2

            if len(layers) == 1:
                layers = layers[0]
                depth = 1
            if isinstance(layers, list) and len(layers) == 0:
                depth = 0
                layers = module

        if level == 0 and isinstance(layers, list) is False:
            layers = [layers]

        return depth, dimensional_layers_num, layers
