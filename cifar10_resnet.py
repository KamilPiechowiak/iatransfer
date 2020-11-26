import torch
from torch import nn

class ResidualBlock(nn.Module):
  def __init__(self, c, downsample=False):
    super(ResidualBlock, self).__init__()
    layers = [
        nn.Conv2d(c, c, kernel_size=3, padding=1),
        nn.BatchNorm2d(c),
        nn.ReLU()
    ]
    if downsample:
      layers+=[
        nn.Conv2d(c, c*2, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(c*2),
        nn.ReLU()
      ]
      self.downsampling_conv = nn.Conv2d(c, c*2, kernel_size=3, padding=1, stride=2)
    else:
      layers+=[
        nn.Conv2d(c, c, kernel_size=3, padding=1),
        nn.BatchNorm2d(c),
        nn.ReLU()
      ]
    self.block = nn.Sequential(*layers)
    self.downsample = downsample
  def forward(self, x):
    if self.downsample:
      return self.block(x) + self.downsampling_conv(x)
    return self.block(x)+x

class Cifar10Resnet(nn.Module):
  def __init__(self, n, num_classes=10, channels=16):
    super(Cifar10Resnet, self).__init__()
    channels = [channels, channels*2, channels*4]
    layers = [
      nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
      nn.BatchNorm2d(channels[0]),
      nn.ReLU()
    ]
    for i, c in enumerate(channels):
      for j in range(n):
        if j == n-1 and i != len(channels)-1:
          layers.append(ResidualBlock(c, downsample=True))
        else:
          layers.append(ResidualBlock(c))

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(channels[-1], num_classes))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
    