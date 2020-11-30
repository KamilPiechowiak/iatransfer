import data
from data import TrainingTuple, DatasetTuple
from cifar10_resnet import Cifar10Resnet
from efficientnet_pytorch import EfficientNet

training_tuples = [
  # TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', DatasetTuple(data.CIFAR10, 32)),
  # TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', DatasetTuple(data.CIFAR10, 32)),
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0'), 'efficientnet-b0', DatasetTuple(data.CIFAR10, 224)),
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1'), 'efficientnet-b1', DatasetTuple(data.CIFAR10, 224))
]

transfer_tuples = [
  (TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', DatasetTuple(data.CIFAR10, 32)),
    'resnet_20'),
  (TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', DatasetTuple(data.CIFAR10, 32)),
    'resnet_14')
]