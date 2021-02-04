from abc import ABC, abstractmethod
from typing import List, Union

import torch.nn as nn


class Standardization(ABC):
    """Base-class for any standardization algorithm detectable from IAT.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def standardize(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        """Standardizes the model to a list of blocks.
        """
        pass

    def __call__(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        """Alias for standardize.
        """
        return self.standardize(module, *args, **kwargs)
