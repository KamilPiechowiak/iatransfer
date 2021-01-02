from typing import Tuple, List

import torch.nn as nn


def flatten_with_blocks(module: nn.Module) -> Tuple[int, int, List[nn.Module]]:
    depth, conv_num, layers = 1, 0, []
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        conv_num = 1
        layers = [module]
    else:
        for child in module.children():
            child_depth, child_conv_num, child_layers = flatten_with_blocks(child)
            if child_depth > 1 or (child_conv_num <= 1):
                layers += child_layers
                conv_num += child_conv_num
                depth = max(depth, child_depth)
            else:
                layers += [child_layers]
                depth = 2
        if len(layers) == 0:
            layers = [module]

    return depth, conv_num, layers
