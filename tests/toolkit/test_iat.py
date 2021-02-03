import unittest

from matching.dp_matching import DPMatching
from standardization.blocks_standardization import BlocksStandardization
from transfer.clip_transfer import ClipTransfer

from iatransfer.toolkit import IAT


class IATTest(unittest.TestCase):

    def _assert_blocks_dp_clip(self, iat: IAT):
        self.assertTrue(isinstance(iat.standardization, BlocksStandardization))
        self.assertTrue(isinstance(iat.matching, DPMatching))
        self.assertTrue(isinstance(iat.transfer, ClipTransfer))

    def test_static(self):
        self.assertIn('BlocksStandardization', IAT._standardization_classes)
        self.assertIn('FlattenStandardization', IAT._standardization_classes)
        self.assertIn('GraphStandardization', IAT._standardization_classes)

    def test_string_init(self):
        iat = IAT('blocks', 'dp', 'clip')

        self._assert_blocks_dp_clip(iat)

    def test_tuple_init(self):
        iat = IAT(('blocks', dict()), ('dp', dict()), ('clip', dict()))

        self._assert_blocks_dp_clip(iat)

    def test_object_init(self):
        iat = IAT((BlocksStandardization(), dict()), (DPMatching(), dict()), (ClipTransfer(), dict()))

        self._assert_blocks_dp_clip(iat)
