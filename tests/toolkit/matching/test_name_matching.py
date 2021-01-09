import unittest

import timm

from iatransfer.toolkit.matching.bipartite_matching import BipartiteMatching
from iatransfer.toolkit.matching.name_matching import NameMatching
from iatransfer.toolkit.matching.nbipartite_matching import NBipartiteMatching


class BipartiteMatchingTest(unittest.TestCase):

    def test_bipartite_matching(self):
        m1 = NameMatching()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        print(m1.sim(amodel, bmodel))