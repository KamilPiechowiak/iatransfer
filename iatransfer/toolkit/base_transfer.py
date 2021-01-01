from abc import ABC, abstractmethod

import torch.nn as nn

from iatransfer.toolkit.transfer.transfer_stats import TransferStats


class Transfer(ABC):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def transfer(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> TransferStats:
        pass

    def __call__(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> TransferStats:
        return self.transfer(from_module, to_module, *args, **kwargs)
