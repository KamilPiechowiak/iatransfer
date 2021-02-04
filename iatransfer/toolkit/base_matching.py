from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch.nn as nn

from iatransfer.toolkit.base_score import Score


class Matching(ABC):
    """Base-class for any matching algorithm detectable from IAT.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def sim(self, from_module: List[Union[nn.Module, List[nn.Module]]],
            to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) -> float:
        """ Calculates the similarity between two standardized models (lists of blocks).

        :return: Float in range [0; 1] representing similarity score.
        """
        raise NotImplementedError()

    @abstractmethod
    def match(self, from_module: List[Union[nn.Module, List[nn.Module]]],
              to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        """ Matches two standardized models (lists of blocks).

        :return: The list of tuples of matched layers.
        """
        pass

    def __call__(self, from_module: List[Union[nn.Module, List[nn.Module]]],
                 to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        """ Alias for 'match'.
        """
        return self.match(from_module, to_module, *args, **kwargs)

    def set_score(self, score: Score) -> None:
        self.score = score