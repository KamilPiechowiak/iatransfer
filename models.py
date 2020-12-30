import data
from data import TrainingTuple, DatasetTuple
from cifar10_resnet import Cifar10Resnet
from efficientnet_pytorch import EfficientNet

training_tuples = [
  # #Small resnet
  # #CIFAR10
  # TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(4), 'resnet_26', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(5), 'resnet_32', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(2, channels=10), 'resnet_narrow_14', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3, channels=10), 'resnet_narrow_20', DatasetTuple(data.CIFAR10, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, channels=10), 'resnet_narrow_26', DatasetTuple(data.CIFAR10, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, channels=10), 'resnet_narrow_32', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(2, channels=24), 'resnet_wide_14', DatasetTuple(data.CIFAR10, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3, channels=24), 'resnet_wide_20', DatasetTuple(data.CIFAR10, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, channels=24), 'resnet_wide_26', DatasetTuple(data.CIFAR10, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, channels=24), 'resnet_wide_32', DatasetTuple(data.CIFAR10, 32), 128),
  # #CIFAR100
  # TrainingTuple(lambda: Cifar10Resnet(2, num_classes=100), 'resnet_14', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3, num_classes=100), 'resnet_20', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(4, num_classes=100), 'resnet_26', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(5, num_classes=100), 'resnet_32', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(2, num_classes=100, channels=10), 'resnet_narrow_14', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3, num_classes=100, channels=10), 'resnet_narrow_20', DatasetTuple(data.CIFAR100, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, num_classes=100, channels=10), 'resnet_narrow_26', DatasetTuple(data.CIFAR100, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, num_classes=100, channels=10), 'resnet_narrow_32', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(2, num_classes=100, channels=24), 'resnet_wide_14', DatasetTuple(data.CIFAR100, 32), 128),
  # TrainingTuple(lambda: Cifar10Resnet(3, num_classes=100, channels=24), 'resnet_wide_20', DatasetTuple(data.CIFAR100, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, num_classes=100, channels=24), 'resnet_wide_26', DatasetTuple(data.CIFAR100, 32), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, num_classes=100, channels=24), 'resnet_wide_32', DatasetTuple(data.CIFAR100, 32), 128),
  # #FASHION_MNIST
  # # TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4), 'resnet_26', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5), 'resnet_32', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(2, channels=10), 'resnet_narrow_14', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(3, channels=10), 'resnet_narrow_20', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, channels=10), 'resnet_narrow_26', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, channels=10), 'resnet_narrow_32', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(2, channels=24), 'resnet_wide_14', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(3, channels=24), 'resnet_wide_20', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(4, channels=24), 'resnet_wide_26', DatasetTuple(data.FASHION_MNIST, 28), 128),
  # # TrainingTuple(lambda: Cifar10Resnet(5, channels=24), 'resnet_wide_32', DatasetTuple(data.FASHION_MNIST, 28), 128),

  #EfficientNet
  #CIFAR10
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0', num_classes=10), 'efficientnet-b0', DatasetTuple(data.CIFAR10, 224), 32),
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1', num_classes=10), 'efficientnet-b1', DatasetTuple(data.CIFAR10, 240), 16),
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b2', num_classes=10), 'efficientnet-b2', DatasetTuple(data.CIFAR10, 260), 16),
  TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b3', num_classes=10), 'efficientnet-b3', DatasetTuple(data.CIFAR10, 300), 8),
  #CIFAR100
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0', num_classes=100), 'efficientnet-b0', DatasetTuple(data.CIFAR100, 224), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1', num_classes=100), 'efficientnet-b1', DatasetTuple(data.CIFAR100, 240), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b2', num_classes=100), 'efficientnet-b2', DatasetTuple(data.CIFAR100, 260), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b3', num_classes=100), 'efficientnet-b3', DatasetTuple(data.CIFAR100, 300), 32),
  #FLOWERS
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0', num_classes=102), 'efficientnet-b0', DatasetTuple(data.FLOWERS, 224), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1', num_classes=102), 'efficientnet-b1', DatasetTuple(data.FLOWERS, 240), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b2', num_classes=102), 'efficientnet-b2', DatasetTuple(data.FLOWERS, 260), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b3', num_classes=102), 'efficientnet-b3', DatasetTuple(data.FLOWERS, 300), 32),
  #AIRCRAFT
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0', num_classes=100), 'efficientnet-b0', DatasetTuple(data.FGVC_AIRCRAFT, 224), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1', num_classes=100), 'efficientnet-b1', DatasetTuple(data.FGVC_AIRCRAFT, 240), 16),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b2', num_classes=100), 'efficientnet-b2', DatasetTuple(data.FGVC_AIRCRAFT, 260), 16),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b3', num_classes=100), 'efficientnet-b3', DatasetTuple(data.FGVC_AIRCRAFT, 300), 8),
  #FOOD
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b0', num_classes=101), 'efficientnet-b0', DatasetTuple(data.FOOD, 224), 32),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b1', num_classes=101), 'efficientnet-b1', DatasetTuple(data.FOOD, 240), 16),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b2', num_classes=101), 'efficientnet-b2', DatasetTuple(data.FOOD, 260), 16),
  # TrainingTuple(lambda: EfficientNet.from_name('efficientnet-b3', num_classes=101), 'efficientnet-b3', DatasetTuple(data.FOOD, 300), 8),
]

transfer_tuples = [
  (TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', DatasetTuple(data.CIFAR10, 32), 128),
    'resnet_20'),
  (TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', DatasetTuple(data.CIFAR10, 32), 128),
    'resnet_14')
]