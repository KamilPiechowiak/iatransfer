from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch.nn as nn


class Standardization(ABC):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def standardize(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        pass

    def __call__(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        return self.standardize(module, *args, **kwargs)
