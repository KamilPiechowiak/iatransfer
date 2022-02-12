import unittest

import timm

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit.transfer.state_dict_transfer import StateDictTransfer
from tests.toolkit.transfer.utils import run_transfer


class StateDictTransferTest(unittest.TestCase):

    def test_on_efficientnet(self):
        return
        run_transfer(
            timm.create_model("efficientnet_b3", pretrained=True),
            timm.create_model("efficientnet_b0", pretrained=True),
            StateDictTransfer()
        )

    def test_on_resnet(self):
        return
        run_transfer(
            Cifar10Resnet(3, no_channels=24),
            Cifar10Resnet(2, no_channels=16),
            StateDictTransfer()
        )