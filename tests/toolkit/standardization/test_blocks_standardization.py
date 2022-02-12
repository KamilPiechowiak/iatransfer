import unittest
from pprint import pprint
from typing import List

import timm
from torch import nn

from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization


class BlocksStandardizationTest(unittest.TestCase):

    def check_arr_length(self, models: List[nn.Module], expected_lengths: List[int]):
        for model, expected_length in zip(models, expected_lengths):
            layers = BlocksStandardization().standardize(model)
            self.assertEqual(len(layers), expected_length)

    def test_efficientnet_pytorch(self):
        from efficientnet_pytorch import EfficientNet
        self.check_arr_length(
            [EfficientNet.from_name(f"efficientnet-b{i}") for i in range(4)],
            [24, 31, 31, 34]
        )

    def test_efficientnet_timm(self):
        self.check_arr_length(
            [timm.create_model(f"efficientnet_b{i}") for i in range(4)],
            [24, 31, 31, 34]
        )

    def test_cifar_resnets(self):
        from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
        self.check_arr_length(
            [Cifar10Resnet(i) for i in range(2, 5)],
            [14, 17, 20]
        )

    def test_mobilenet(self):
        from torchvision import models
        self.check_arr_length(
            [models.MobileNetV2()],
            [25]
        )

    def test_model(self):
        model = timm.create_model("regnetx_004")
        pprint(model)
        bs = BlocksStandardization()
        layers = bs.standardize(model)
        print(len(layers))
        pprint(layers)

    def test_on_resnet(self):
        from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
        model = Cifar10Resnet(2)
        pprint(model)
        layers = BlocksStandardization().standardize(model)
        pprint(layers)