import unittest

import timm

from iatransfer.toolkit import IAT


class RandomMatchingTest(unittest.TestCase):

    def test_random_matching(self):
        m1 = IAT(matching='random')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.run(amodel, bmodel))

    def test_output_matching(self):
        m1 = IAT(matching='random')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.run(amodel, bmodel))
