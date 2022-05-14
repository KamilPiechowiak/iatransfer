from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn

from iatransfer.toolkit.transfer.transfer_stats import TransferStats


class Transfer(ABC):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def transfer(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) \
            -> TransferStats:
        with torch.no_grad():
            for matching in matched_tensors:
                if isinstance(matching, list):
                    self.transfer(matching)
                elif matching[0] is not None and matching[1] is not None:
                    self.transfer_layer(matching[0].weight, matching[1].weight)
                    self.transfer_layer(matching[0].bias, matching[1].bias)
                    self.postprocess(matching[1])
        return TransferStats()

    @abstractmethod
    def transfer_layer(self, tensor_from: nn.Module, tensor_to: nn.Module, *args, **kwargs) \
            -> TransferStats:
        pass

    @abstractmethod
    def postprocess(self, layer: nn.Module) -> TransferStats:
        pass

    def __call__(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) \
            -> TransferStats:
        return self.transfer(matched_tensors, *args, **kwargs)
