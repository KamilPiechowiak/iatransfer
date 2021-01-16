import unittest
from torch import nn
import timm
from pprint import pprint

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit.transfer.magnitude_transfer import MagnitudeTransfer
from tests.toolkit.transfer.utils import run_transfer


class MagnitudeTransferTest(unittest.TestCase):

    def test_on_efficientnet(self):
        run_transfer(
            timm.create_model("efficientnet_b3"),
            timm.create_model("efficientnet_b0"),
            MagnitudeTransfer()
        )

    def test_on_resnet(self):
        run_transfer(
            Cifar10Resnet(3, no_channels=24),
            Cifar10Resnet(2, no_channels=16),
            MagnitudeTransfer()
        )