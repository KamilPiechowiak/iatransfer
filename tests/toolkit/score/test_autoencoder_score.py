import unittest

import timm

from iatransfer.toolkit.score.autoencoder_score import AutoEncoderScore


class AutoEncoderScoreTest(unittest.TestCase):

    def test_on_efficientnet(self):
        a = AutoEncoderScore()
        from_model = timm.create_model("efficientnet_b3")
        to_model = timm.create_model("efficientnet_b0")
        a.precompute_scores(from_model, to_model)
        print(a.score(from_model.conv_stem, to_model.conv_stem))
        print(a.score(from_model.conv_head, to_model.conv_head))
        print(a.score(from_model.conv_stem, to_model.conv_head))
        print(a.score(from_model.conv_head, to_model.conv_stem))
