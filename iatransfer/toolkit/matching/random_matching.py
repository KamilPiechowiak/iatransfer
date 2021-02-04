import random
from typing import Tuple, List, Union

import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching


class RandomMatching(Matching):
    """Random matching algorithm for IAT. Used for baseline.
    """

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