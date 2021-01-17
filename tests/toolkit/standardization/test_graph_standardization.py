import unittest
from pprint import pprint

import timm

from iatransfer.toolkit.standardization.graph_standardization import GraphStandardization


class GraphStandardizationTest(unittest.TestCase):

    def test_model(self):
        model = timm.create_model("regnetx_002")
        bs = GraphStandardization()
        layers = bs.standardize(model)
        print(len(layers))
        pprint(layers)

    def show_graph(self, model, path="__tli_debug"):
        import os
        # FIXME: warning about 'torchviz'
        x = torch.randn(32, 3, 32, 32)
        v1, _, _, _, _ = GraphStandardization().make_dot(model(x), params=dict(model.named_parameters()))
        v1.render(filename=path + "__v1")
        os.system(f"rm {path}__v1")
        print("saved to file")
