from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, c, downsample: bool = False) -> None:
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU()
        ]
        if downsample:
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c * 2),
                nn.ReLU()
            ]
            self.downsampling_conv = nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2)
        else:
            layers += [
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU()
            ]
        self.block = nn.Sequential(*layers)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            return self.block(x) + self.downsampling_conv(x)
        return self.block(x) + x


class Cifar10Resnet(nn.Module):
    def __init__(self, no_blocks: int, no_classes: int = 10, no_channels: int = 16, no_input_channels: int = 3) -> None:
        super(Cifar10Resnet, self).__init__()
        no_channels = [no_channels, no_channels * 2, no_channels * 4]
        layers = [
            nn.Conv2d(no_input_channels, no_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(no_channels[0]),
            nn.ReLU()
        ]
        for i, c in enumerate(no_channels):
            for j in range(no_blocks):
                if j == no_blocks - 1 and i != len(no_channels) - 1:
                    layers.append(ResidualBlock(c, downsample=True))
                else:
                    layers.append(ResidualBlock(c))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(no_channels[-1], no_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
