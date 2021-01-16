import unittest

from pprint import pprint

from iatransfer.toolkit.transfer_methods_factory import TransferMethodsFactory


class TransferMethodsFactoryTest(unittest.TestCase):

    def test_init(self):
        t = TransferMethodsFactory()

        pprint(t.transfer_classes)
        pprint(t.matching_classes)
        pprint(t.standardization_classes)
