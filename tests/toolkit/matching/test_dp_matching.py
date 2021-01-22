import unittest
from pprint import pprint

import timm

from iatransfer.research.models.cifar10_resnet import Cifar10Resnet
from iatransfer.toolkit import IAT
from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.standardization.graph_standardization import GraphStandardization


class DPMatchingTest(unittest.TestCase):

    def test_dp_matching_on_efficientnet(self):
        m1 = IAT(matching='dp')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.sim(amodel, bmodel))

    def test_dp_matching_on_resnet(self):
        m1 = IAT(matching='dp')

        amodel = Cifar10Resnet(3, no_channels=24)
        bmodel = Cifar10Resnet(2, no_channels=16)

        pprint(m1.sim(amodel, bmodel))

    def test_dp_matching_with_graph_standardization(self):
        m1 = IAT(matching='dp', standardization='blocks')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))
