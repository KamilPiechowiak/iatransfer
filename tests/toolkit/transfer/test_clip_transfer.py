import unittest

import timm

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit import IAT
from tests.toolkit.transfer.utils import run_transfer


class ClipTransferTest(unittest.TestCase):

    def test_on_efficientnet(self):
        run_transfer(
            timm.create_model("efficientnet_b3"),
            timm.create_model("efficientnet_b0"),
            IAT(transfer='clip')
        )

    def test_on_resnet(self):
        run_transfer(
            Cifar10Resnet(3, no_channels=24),
            Cifar10Resnet(2, no_channels=16),
            IAT(transfer='clip')
        )