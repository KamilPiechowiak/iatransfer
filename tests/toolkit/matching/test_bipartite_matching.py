import unittest
from pprint import pprint

import timm

from iatransfer.toolkit import IAT

class BipartiteMatchingTest(unittest.TestCase):

    def test_bipartite_matching(self):
        m1 = IAT(matching='bipartite')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))

    def test_output_matching(self):
        m1 = IAT(matching='bipartite')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1(amodel, bmodel))
