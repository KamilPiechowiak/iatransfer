from abc import ABC, abstractmethod

import torch.nn as nn


class Score(ABC):
    """Base-class for any scoring algorithm detectable from IAT.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def score(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> float:
        """Calculates the float score between two layers.
        """
        pass

    def precompute_scores(self, from_model: nn.Module, to_model: nn.Module, *args, **kwargs) \
            -> float:
        pass

    def __call__(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> float:
        """Alias for 'score'.
        """
        return self.score(from_module, to_module, *args, **kwargs)
