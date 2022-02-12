import unittest

import timm

from iatransfer.toolkit.tli import show_graph


class TransferTLI(unittest.TestCase):

    def test_visualisation(self) -> None:
        model = timm.create_model("regnety_004")
        show_graph(model, path="regnety_004")
        model = timm.create_model("regnetx_004")
        show_graph(model, path="regnetx_004")