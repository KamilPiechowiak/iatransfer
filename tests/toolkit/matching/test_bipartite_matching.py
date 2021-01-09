import unittest

import timm

from iatransfer.toolkit.matching.bipartite_matching import BipartiteMatching


class BipartiteMatchingTest(unittest.TestCase):

    def test_bipartite_matching(self):
        m1 = BipartiteMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))