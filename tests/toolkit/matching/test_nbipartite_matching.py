import unittest
from pprint import pprint

import timm

from iatransfer.toolkit import IAT


class NBipartiteMatchingTest(unittest.TestCase):

    def test_bipartite_matching(self):
        m1 = IAT(matching='n_bipartite')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.run(amodel, bmodel))

    def test_output_matching(self):
        m1 = IAT(matching='n_bipartite')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.run(amodel, bmodel))
