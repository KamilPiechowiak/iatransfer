import unittest

from iatransfer.toolkit import IAT
from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization
from iatransfer.toolkit.transfer.clip_transfer import ClipTransfer
from iatransfer.utils.subclass_utils import get_subclasses


class IATTest(unittest.TestCase):

    def _assert_blocks_dp_clip(self, iat: IAT):
        self.assertEquals(iat.standardization.__class__.__name__, 'BlocksStandardization')
        self.assertEquals(iat.matching.__class__.__name__, 'DPMatching')
        self.assertEquals(iat.transfer.__class__.__name__, 'ClipTransfer')

    def test_static(self):
        self.assertIn('BlocksStandardization', get_subclasses(Standardization))
        self.assertIn('FlattenStandardization', get_subclasses(Standardization))
        self.assertIn('GraphStandardization', get_subclasses(Standardization))

    def test_string_init(self):
        iat = IAT('blocks', 'dp', 'clip')

        self._assert_blocks_dp_clip(iat)

    def test_tuple_init(self):
        iat = IAT(('blocks', dict()), ('dp', dict()), ('clip', dict()))

        self._assert_blocks_dp_clip(iat)

    def test_object_init(self):
        iat = IAT((BlocksStandardization(), dict()), (DPMatching(), dict()), (ClipTransfer(), dict()))

        self._assert_blocks_dp_clip(iat)

    def test_plugin(self):
        import random
        import timm
        from typing import List, Union, Tuple
        from torch import nn
        from iatransfer.toolkit.base_matching import Matching

        class PluginMatching(Matching):
            def match(self, from_module: List[Union[nn.Module, List[nn.Module]]],
                      to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
                    -> List[Union[Tuple[nn.Module, nn.Module], List[Tuple[nn.Module, nn.Module]]]]:
                matched = self._match_models(from_module, to_module)
                return matched

            def sim(self, from_module: List[Union[nn.Module, List[nn.Module]]],
                    to_module: List[Union[nn.Module, List[nn.Module]]]) -> float:
                return random.random()

            def _get_layers_buckets(self, module: nn.Module):
                classes = [
                    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.Linear
                ]
                buckets = [[] for _ in range(len(classes))]
                for layer in module:
                    for i, clazz in enumerate(classes):
                        if isinstance(layer, clazz):
                            buckets[i].append(layer)
                for i in range(len(buckets)):
                    random.shuffle(buckets[i])
                return buckets

            def _match_models(self, flat_from_module: List[nn.Module],
                              flat_to_module: List[nn.Module]) \
                    -> List[Tuple[nn.Module, nn.Module]]:
                teacher_buckets = self._get_layers_buckets(flat_from_module)
                student_buckets = self._get_layers_buckets(flat_to_module)

                matched = []
                for teacher_bucket, student_bucket in zip(teacher_buckets, student_buckets):
                    for teacher_layer, student_layer in zip(teacher_bucket, student_bucket):
                        matched.append((teacher_layer, student_layer))
                return matched

        iat = IAT(matching='plugin')

        amodel = timm.create_model("efficientnet_b3")
        bmodel = timm.create_model("efficientnet_b0")

        iat(amodel, bmodel)
