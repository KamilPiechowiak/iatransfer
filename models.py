from data import TrainingTuple, cifar_train, cifar_val
from cifar10_resnet import Cifar10Resnet

training_tuples = [
  TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', cifar_train, cifar_val),
  TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', cifar_train, cifar_val)
]

transfer_tuples = [
  (TrainingTuple(lambda: Cifar10Resnet(2), 'resnet_14', cifar_train, cifar_val),
    'resnet_20'),
  (TrainingTuple(lambda: Cifar10Resnet(3), 'resnet_20', cifar_train, cifar_val),
    'resnet_14')
]