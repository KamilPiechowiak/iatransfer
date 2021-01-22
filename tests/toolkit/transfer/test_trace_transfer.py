import unittest

import timm

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit import IAT
from tests.toolkit.transfer.utils import run_transfer


class TraceTransferTest(unittest.TestCase):

    def test_on_efficientnet(self):
        run_transfer(
            timm.create_model("efficientnet_b3"),
            timm.create_model("efficientnet_b0"),
            IAT(transfer='trace')
        )

    def test_on_resnet(self):
        run_transfer(
            Cifar10Resnet(3, no_channels=24),
            Cifar10Resnet(2, no_channels=16),
            IAT(transfer='trace')
        )

    def test_on_mixnet_s_to_efficientnet_b1(self):
        run_transfer(
            timm.create_model("mixnet_s"),
            timm.create_model("efficientnet_b1"),
            IAT(transfer='trace')
        )