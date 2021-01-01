from abc import ABC, abstractmethod
from typing import List, Tuple

import torch.nn as nn


class Matching(ABC):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def match(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> List[Tuple[nn.Module, List[nn.Module]]]:
        pass

    def __call__(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> List[Tuple[nn.Module, List[nn.Module]]]:
        return self.match(from_module, to_module, *args, **kwargs)
