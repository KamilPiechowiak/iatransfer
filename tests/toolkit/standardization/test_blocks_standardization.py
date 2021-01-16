import unittest
from typing import List
from torch import nn
import timm
from pprint import pprint

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
        model = timm.create_model("regnetx_002")
        bs = BlocksStandardization()
        layers = bs.standardize(model)
        print(len(layers))
        pprint(layers)