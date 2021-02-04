import unittest

import timm

from iatransfer.toolkit import IAT
from iatransfer.toolkit.transfer.transfer_stats import TransferStats


class TransferStatsTest(unittest.TestCase):

    def test_transfer_stats(self) -> None:
        iat = IAT()

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        stats = iat(amodel, bmodel)
        self.assertEqual(stats, TransferStats(matched_from=131, matched_to=131, left_from=159, left_to=51, all_from=290,
                                              all_to=182))
