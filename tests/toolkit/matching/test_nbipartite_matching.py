import unittest

import timm
from pprint import pprint

from iatransfer.toolkit.matching.bipartite_matching import BipartiteMatching
from iatransfer.toolkit.matching.nbipartite_matching import NBipartiteMatching


class BipartiteMatchingTest(unittest.TestCase):

    def test_bipartite_matching(self):
        m1 = NBipartiteMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))

    def test_output_matching(self):
        m1 = BipartiteMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        pprint(m1.match(amodel, bmodel))