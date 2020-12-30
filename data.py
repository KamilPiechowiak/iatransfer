from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from typing import NamedTuple, Callable
from pretrain_flags import FLAGS
from custom_datasets import Flowers102, FGVCAircraft, Food101
import PIL
from enum import Enum

class DatasetTuple(NamedTuple):
  name: int
  resolution: int

class TrainingTuple(NamedTuple):
  model: Callable[[], nn.Module]
  name: str
  dataset_tuple: DatasetTuple
  batch_size: int

FASHION_MNIST = 1
CIFAR10 = 2
CIFAR100 = 3
FLOWERS = 4
FGVC_AIRCRAFT = 5
FOOD = 6

def get_dataset_name(dataset_tuple: DatasetTuple) -> str:
  # print(dataset_tuple)
  return ["", "FASHION_MNIST", "CIFAR10", "CIFAR100", "FLOWERS", "FGVC_AIRCRAFT", "FOOD"][dataset_tuple.name]

def get_dataset(dataset_tuple: DatasetTuple):

  def prepare_dataset(name: int, train: bool, resolution: int):
    if name == FASHION_MNIST:
      normalize = transforms.Normalize(mean=0.2860, std=0.3530)
    else:
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
      if name == FASHION_MNIST:
        scale = (0.8, 1)
      else:
        scale = (0.5, 1)
      preprocess = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
      ])
    else:
      preprocess = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize
      ])
    path = FLAGS['datasets_path']
    return {
      FASHION_MNIST: lambda: datasets.FashionMNIST(f"{path}/fashion_mnist", train=train, transform=preprocess, download=True),
      CIFAR10: lambda: datasets.CIFAR10(f"{path}/cifar10", train=train, transform=preprocess, download=True),
      CIFAR100: lambda: datasets.CIFAR100(f"{path}/cifar100", train=train, transform=preprocess, download=True),
      FLOWERS: lambda: Flowers102(f"{path}", split='train' if train else 'val', transform=preprocess, download=True),
      FGVC_AIRCRAFT: lambda: FGVCAircraft(f"{path}", split='train' if train else 'test', transform=preprocess, download=True),
      FOOD: lambda: Food101(f"{path}", train, transform=preprocess, download=True),
    }[name]
  return prepare_dataset(dataset_tuple.name, True, dataset_tuple.resolution), prepare_dataset(dataset_tuple.name, False, dataset_tuple.resolution)


# preprocess for EfficientNet - Resize(224)!!!
# preprocess = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(224, interpolation=PIL.Image.NEAREST),
#     transforms.Pad(10),
#     transforms.RandomCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# preprocess for small resnet
# preprocess = transforms.Compose([
#   transforms.Pad(4),
#   transforms.RandomCrop(32),
#   transforms.RandomHorizontalFlip(),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# cifar_train = lambda: datasets.CIFAR10(FLAGS['datasets_path'], train=True, download=True, transform=preprocess)
# cifar_val = lambda: datasets.CIFAR10(FLAGS['datasets_path'], train=False, download=True, transform=preprocess)