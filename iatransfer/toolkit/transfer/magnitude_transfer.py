from typing import List, Tuple, Union

import torch
import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_transfer import Transfer
from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.transfer.transfer_stats import TransferStats
from iatransfer.toolkit.transfer.transfer_stats import get_absmeans


class MagnitudeTransfer(Transfer):
    """
        :param reverse_priority: if set to True, choose channels with smallest weights' sum
        :param input_constrained_by_output: if set to True, choose input channels considering only already chosen output channels. If set to False, choose input channels taking into account all weights.
    """
    def __init__(self, matching_strategy: Matching = DPMatching(), reverse_priority = False, input_constrained_by_output = True, **kwargs) -> None:
        self.matching_strategy = matching_strategy
        if reverse_priority:
            self.sgn = +1
        else:
            self.sgn = -1
        self.cut_from_tensor = input_constrained_by_output
        super().__init__(**kwargs)

    def transfer(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) -> TransferStats:
        matched = self.matching_strategy(from_module, to_module)
        self._transfer(matched)
        return TransferStats()

    def update_slice(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor, idx: int, from_slices: List[Union[List[int], slice]], to_slices: List[Union[List[int], slice]]) -> None:
        dims = [i for i in range(len(from_tensor.shape))]
        dims.remove(idx)
        if from_tensor.shape[idx] > to_tensor.shape[idx]:
            if len(dims) == 0:
                out_channels = list(enumerate(from_tensor))
            else: 
                out_channels = list(enumerate(from_tensor.abs().sum(dim=dims)))
            out_channels.sort(key=lambda w : self.sgn*w[1])
            c = to_tensor.shape[idx]
            from_ids = sorted([w[0] for w in out_channels[:c]])
            from_slices.append(from_ids)
            to_slices.append(slice(0, c))
            if self.cut_from_tensor:
                from_tensor = torch.index_select(from_tensor, idx, torch.tensor(from_ids))
        else:
            from_slices.append(slice(0, from_tensor.shape[idx]))
            to_slices.append(slice(0, from_tensor.shape[idx]))

    def get_slices(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor) -> Tuple[Tuple[Union[List[int], slice]]]:
        if from_tensor is None or to_tensor is None:
            return None
        n = len(from_tensor.shape)
        from_slices, to_slices = [], []
        self.update_slice(from_tensor, to_tensor, 0, from_slices, to_slices)
        if n > 1:
            self.update_slice(from_tensor, to_tensor, 1, from_slices, to_slices)

        for a, b in zip(from_tensor.shape[2:], to_tensor.shape[2:]):
            if a < b:
                from_slices.append(slice(0, a))
                to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
                to_slices.append(slice(0, b))
            else:
                from_slices.append(slice(0, a))
                to_slices.append(slice(0, b))
        total_unsqueeze = 0
        for i in range(len(from_slices)-1, -1, -1):
            if isinstance(from_slices[i], list):
                from_slices[i] = torch.tensor(from_slices[i])
                for _ in range(total_unsqueeze):
                    from_slices[i] = from_slices[i].unsqueeze(-1)
                total_unsqueeze+=1
        return tuple(to_slices), tuple(from_slices)

    def _transfer(self, matched: List[Tuple[nn.Module, nn.Module]]) -> None:
        with torch.no_grad():
            for matching in matched:
                if isinstance(matching, list):
                    self._transfer(matching)
                elif matching[0] is not None and matching[1] is not None:
                    to_slices, from_slices = self.get_slices(matching[0].weight, matching[1].weight)
                    # import sys
                    # print(to_slices, "<=", from_slices, file=sys.stderr)
                    if matching[1].weight is not None and matching[0].weight is not None:
                        # print(matching[1].weight[to_slices].shape)
                        # print(matching[0].weight.shape)
                        matching[1].weight[to_slices] = matching[0].weight[from_slices]
                    if matching[1].bias is not None and matching[0].bias is not None:
                        # print(matching[0].bias.shape)
                        # print(matching[1].bias.shape)
                        to_ids = to_slices[0]
                        from_ids = from_slices[0]
                        if isinstance(to_ids, torch.Tensor):
                            to_ids = to_ids.flatten()
                        if isinstance(from_ids, torch.Tensor):
                            from_ids = from_ids.flatten()
                        matching[1].bias[to_ids] = matching[0].bias[from_ids]


if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet

    amodel = EfficientNet.from_pretrained('efficientnet-b0')
    bmodel = EfficientNet.from_pretrained('efficientnet-b3')

    stats_before = get_absmeans(amodel)
    MagnitudeTransfer()(bmodel, amodel)
    stats_after = get_absmeans(amodel)
    print('\n'.join(
        [str((x, y)) for x, y in zip(stats_before, stats_after)]
    ))
