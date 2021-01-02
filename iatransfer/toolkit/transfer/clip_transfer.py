from typing import List, Tuple

import torch
import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_transfer import Transfer
from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.transfer.transfer_stats import TransferStats
from iatransfer.toolkit.transfer.transfer_stats import get_absmeans


class ClipTransfer(Transfer):

    def __init__(self, matching_strategy: Matching = DPMatching(), **kwargs) -> None:
        self.matching_strategy = matching_strategy
        super().__init__(**kwargs)

    def transfer(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) -> TransferStats:
        matched = self.matching_strategy(from_module, to_module)
        self._transfer(matched)
        return TransferStats()

    def _transfer_tensors(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor) -> None:
        if from_tensor is None or to_tensor is None:
            return
        from_slices, to_slices = [], []
        for a, b in zip(from_tensor.shape, to_tensor.shape):
            if a < b:
                from_slices.append(slice(0, a))
                to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
                to_slices.append(slice(0, b))
            else:
                from_slices.append(slice(0, a))
                to_slices.append(slice(0, b))
        to_tensor[tuple(to_slices)] = from_tensor[tuple(from_slices)]

    def _transfer(self, matched: List[Tuple[nn.Module, nn.Module]]) -> None:
        with torch.no_grad():
            for matching in matched:
                if isinstance(matching, list):
                    self._transfer(matching)
                elif matching[0] is not None and matching[1] is not None:
                    self._transfer_tensors(matching[0].weight, matching[1].weight)
                    self._transfer_tensors(matching[0].bias, matching[1].bias)


if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet

    amodel = EfficientNet.from_pretrained('efficientnet-b0')
    bmodel = EfficientNet.from_pretrained('efficientnet-b3')

    stats_before = get_absmeans(bmodel)
    ClipTransfer()(amodel, bmodel)
    stats_after = get_absmeans(bmodel)
    print('\n'.join(
        [str((x, y)) for x, y in zip(stats_before, stats_after)]
    ))
