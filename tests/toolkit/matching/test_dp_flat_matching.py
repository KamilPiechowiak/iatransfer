import unittest
from pprint import pprint

import timm

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit import IAT


class DPFlatMatchingTest(unittest.TestCase):

    def test_dp_flat_matching_on_efficientnet(self):
        m1 = IAT(matching='dp_flat')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.run(amodel, bmodel))

    def test_dp_flat_matching_on_resnet(self):
        m1 = IAT(matching='dp_flat')

        amodel = Cifar10Resnet(3, no_channels=24)
        bmodel = Cifar10Resnet(2, no_channels=16)

        pprint(m1.run(amodel, bmodel))
