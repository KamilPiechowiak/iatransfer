import unittest

from iatransfer.toolkit import IAT
from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization
from iatransfer.toolkit.transfer.clip_transfer import ClipTransfer


class IATTest(unittest.TestCase):

    def _assert_blocks_dp_clip(self, iat: IAT):
        self.assertEquals(iat.standardization.__class__.__name__, 'BlocksStandardization')
        self.assertEquals(iat.matching.__class__.__name__, 'DPMatching')
        self.assertEquals(iat.transfer.__class__.__name__, 'ClipTransfer')

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
