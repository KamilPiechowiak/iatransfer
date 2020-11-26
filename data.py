from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from typing import NamedTuple, Callable

class TrainingTuple(NamedTuple):
  model: Callable[[], nn.Module]
  name: str
  train_dataset: Callable[[], Dataset]
  val_dataset: Callable[[], Dataset]

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
preprocess = transforms.Compose([
  transforms.Pad(4),
  transforms.RandomCrop(32),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
cifar_train = lambda: datasets.CIFAR10("/content/data/", train=True, download=True, transform=preprocess)
cifar_val = lambda: datasets.CIFAR10("/content/data/", train=False, download=True, transform=preprocess)