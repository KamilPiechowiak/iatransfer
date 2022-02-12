from typing import NamedTuple, Callable, Dict, Optional

import timm
import torch
from torch import nn
from torchvision import transforms, datasets

from iatransfer.research.data import Flowers102, FGVCAircraft, Food101
from iatransfer.research.models.cifar10_resnet import Cifar10Resnet


class DatasetTuple(NamedTuple):
    name: str
    resolution: int


class TrainingTuple(NamedTuple):
    model: Callable[[], nn.Module]
    name: str
    dataset_tuple: DatasetTuple
    batch_size: int
    lr: Optional[float]

    @staticmethod
    def from_json(json: Dict) -> 'TrainingTuple':
        if json["model"]["supplier"] == "Cifar10Resnet":
            supplier = Cifar10Resnet
        else:
            supplier = timm.create_model

        def model():
            return supplier(*json["model"]["args"], **json["model"]["kwargs"])

        lr = None
        if "lr" in json.keys():
            lr = float(json["lr"])

        return TrainingTuple(
            model,
            json["model"]["name"],
            DatasetTuple(json["dataset"]["name"], json["dataset"]["resolution"]),
            json["batch_size"],
            lr
        )


def get_dataset(dataset_tuple: DatasetTuple, FLAGS: Dict):
    def prepare_dataset(name: str, train: bool, resolution: int):
        stack = transforms.Lambda(lambda img: torch.cat([img, img, img], axis=0) if img.shape[0] == 1 else img)
        if name == 'FASHION_MNIST':
            normalize = transforms.Normalize(mean=0.2860, std=0.3530)
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            if name == 'FASHION_MNIST':
                scale = (0.8, 1)
            # elif name.startswith("CIFAR"):
            #     scale = (0.4, 1)
            elif resolution < 100:
                scale = (0.5, 1)
            else:
                scale = (0.1, 1)
            preprocess = transforms.Compose([
                transforms.RandomResizedCrop(resolution, scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                stack,
                normalize
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                stack,
                normalize
            ])
        path = FLAGS['datasets_path']
        return {
            'FASHION_MNIST': lambda: datasets.FashionMNIST(f'{path}/fashion_mnist', train=train, transform=preprocess,
                                                           download=True),
            'CIFAR10': lambda: datasets.CIFAR10(f'{path}/cifar10', train=train, transform=preprocess, download=True),
            'CIFAR100': lambda: datasets.CIFAR100(f'{path}/cifar100', train=train, transform=preprocess, download=True),
            'FLOWERS': lambda: Flowers102(f'{path}', split='train' if train else 'val', transform=preprocess,
                                          download=True),
            'FGVC_AIRCRAFT': lambda: FGVCAircraft(f'{path}', split='train' if train else 'test', transform=preprocess,
                                                  download=True),
            'FOOD': lambda: Food101(f'{path}', train, transform=preprocess, download=True),
        }[name]

    return prepare_dataset(dataset_tuple.name,
                           True,
                           dataset_tuple.resolution)(), \
        prepare_dataset(dataset_tuple.name,
                        False,
                        dataset_tuple.resolution)()
