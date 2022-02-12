from typing import Dict

from torch import nn
import timm

from iatransfer.researchput.models.cifar10_resnet import Cifar10Resnet

def get_model(config: Dict, init: str) -> nn.Module:
    supplier_name = config["supplier"]
    supplier = None
    args = config["args"]
    kwargs = config["kwargs"]
    if init == "pretrained":
        kwargs.update({"pretrained": True})
    if  supplier_name == "Cifar10Resnet":
        supplier = Cifar10Resnet
    elif supplier_name == "timm.create_model":
        supplier = timm.create_model
    else:
        raise RuntimeError(f"No such model supplier: {supplier_name}")
    return supplier(*args, **kwargs)