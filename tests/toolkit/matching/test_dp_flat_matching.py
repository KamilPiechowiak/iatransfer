import unittest

import timm
from pprint import pprint

from iatransfer.toolkit.matching.dp_flat_matching import DPFlatMatching
from iatransfer.research.models.cifar10_resnet import Cifar10Resnet

class DPFlatMatchingTest(unittest.TestCase):

    def test_dp_flat_matching_on_efficientnet(self):
        m1 = DPFlatMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.match(amodel, bmodel))

    def test_dp_flat_matching_on_resnet(self):
        m1 = DPFlatMatching()

        amodel = Cifar10Resnet(3, no_channels=24)
        bmodel = Cifar10Resnet(2, no_channels=16)

        pprint(m1.match(amodel, bmodel))
