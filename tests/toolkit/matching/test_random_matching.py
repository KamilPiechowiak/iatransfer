import unittest
from pprint import pprint

import timm

from iatransfer.toolkit.matching.random_matching import RandomMatching


class RandomMatchingTest(unittest.TestCase):

    def test_random_matching(self):
        m1 = RandomMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))

    def test_output_matching(self):
        m1 = RandomMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.match(amodel, bmodel))
