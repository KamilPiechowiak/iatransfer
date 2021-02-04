from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class Transfer(ABC):
    """Base-class for any tensor transfer algorithm detectable from IAT.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def transfer(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) -> None:
        """Invokes the transfer procedure for a list of matched tensors.
        """
        with torch.no_grad():
            for matching in matched_tensors:
                if isinstance(matching, list):
                    self.transfer(matching)
                elif matching[0] is not None and matching[1] is not None:
                    self.transfer_layer(matching[0].weight, matching[1].weight)
                    self.transfer_layer(matching[0].bias, matching[1].bias)

    @abstractmethod
    def transfer_layer(self, tensor_from: nn.Module, tensor_to: nn.Module, *args, **kwargs) -> None:
        """Transfers the matched layers. Changes the contents of matched tensors in model_to.
        """
        pass

    def __call__(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) -> None:
        """Alias for 'transfer'.
        """
        self.transfer(matched_tensors, *args, **kwargs)
