from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch.nn as nn


class Matching(ABC):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def sim(self, from_module: List[Union[nn.Module, List[nn.Module]]], to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) -> float:
        raise NotImplementedError()

    @abstractmethod
    def match(self, from_module: List[Union[nn.Module, List[nn.Module]]], to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        pass

    def __call__(self, from_module: List[Union[nn.Module, List[nn.Module]]], to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        return self.match(from_module, to_module, *args, **kwargs)
