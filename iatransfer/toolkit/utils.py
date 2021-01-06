from typing import Tuple, List

import torch.nn as nn


def flatten_with_blocks(module: nn.Module, level: int = 0) -> Tuple[int, int, List[nn.Module]]:
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
            next_level = level+1
        for child in module.children():
            child_depth, child_dimensional_layers_num, child_layers = flatten_with_blocks(child, next_level)
            dimensional_layers_num += child_dimensional_layers_num 
            if child_depth > 1 or (isinstance(child_layers, list) and child_dimensional_layers_num < 2):
                for child_layer in child_layers:
                    if isinstance(child_layer, list):
                        blocks+=1
                    else:
                        single_layers+=1
                    layers+=[child_layer]
            else:
                if isinstance(child_layers, list):
                    blocks+=1
                else:
                    single_layers+=1
                layers+=[child_layers]
        
        # print("before: ", layers)
        
        if single_layers > blocks and level > 0:
            new_layers = []
            for child_layers in layers:
                if isinstance(child_layers, list):
                    new_layers+=child_layers
                else:
                    new_layers+=[child_layers]
            layers = new_layers
            depth = 1
        else:
            depth = 2

        if len(layers) == 1:
            # print(layers)
            layers = layers[0]
            depth = 1
        if isinstance(layers, list) and len(layers) == 0:
            # print(module)
            depth = 0
            layers = module

        # print("after: ", layers)
    
    if level == 0 and isinstance(layers, list) == False:
        layers = [layers]

    return depth, dimensional_layers_num, layers

def flatten(module: nn.Module) ->  List[nn.Module]:
    classes = [
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Linear
    ]
    if any([isinstance(module, clazz) for clazz in classes]):
        return [module]
    
    layers = []
    for child in module.children():
        layers += flatten(child)
    if len(layers) == 0:
        layers = [module]
    return layers

if __name__ == "__main__":
    from efficientnet_pytorch import EfficientNet
    import timm
    from torchvision import models
    from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
    from pprint import pprint

    def test(model):
        print(model)
        _, _, layers = flatten_with_blocks(model)
        print(len(layers))
        # pprint(layers)

    # test(Cifar10Resnet(2, 10, 10))
    # test(timm.create_model("efficientnet_b"))
    # test(EfficientNet.from_name("efficientnet-b3"))
    # test(models.MobileNetV2())
    pprint(flatten(Cifar10Resnet(2, 10, 10)))