from typing import List, Tuple

import torch
import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.transfer._matched_transfer import MatchedTransfer
from iatransfer.toolkit.transfer.transfer_stats import TransferStats
from iatransfer.toolkit.transfer.transfer_stats import get_absmeans


class FullTransfer(MatchedTransfer):

    def __init__(self, **kwargs) -> None:
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
                # from_slices.append(slice(0, a))
                # to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
                ids = torch.randint(a, (b,))
                ids[slice((b - a) // 2, -((b - a + 1) // 2))] = torch.arange(a)
                from_slices.append(ids)
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
            else:
                from_slices.append(slice(0, a))
            to_slices.append(slice(0, b))

        total_unsqueeze = 0
        for i in range(len(from_slices)-1, -1, -1):
            if isinstance(from_slices[i], torch.Tensor):
                for _ in range(total_unsqueeze):
                    from_slices[i] = from_slices[i].unsqueeze(-1)
                total_unsqueeze+=1

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
    import timm
    from efficientnet_pytorch import EfficientNet
    from iatransfer.research.models.cifar10_resnet import Cifar10Resnet

    # amodel = EfficientNet.from_pretrained('efficientnet-b3')
    # amodel = EfficientNet.from_pretrained('efficientnet-b0')
    # bmodel = EfficientNet.from_pretrained('efficientnet-b0')
    bmodel = timm.create_model('efficientnet_b0', pretrained=True)
    # amodel = timm.create_model('efficientnet_b0', pretrained=True)
    amodel = timm.create_model('efficientnet_b3', pretrained=True)
    # amodel = Cifar10Resnet(3, no_channels=24)
    # bmodel = Cifar10Resnet(2, no_channels=16)

    stats_before = get_absmeans(bmodel)
    FullTransfer()(amodel, bmodel)
    stats_after = get_absmeans(bmodel)
    print('\n'.join(
        [str((x, y)) for x, y in zip(stats_before, stats_after)]
    ))
