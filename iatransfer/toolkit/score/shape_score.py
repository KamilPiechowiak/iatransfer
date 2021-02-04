from torch import nn

from iatransfer.toolkit.base_score import Score


class ShapeScore(Score):
    """Shape based score algorithm for IAT.
    """

    def score(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> float:
        score = 1
        for x, y in zip(from_module.weight.shape, to_module.weight.shape):
            score *= min(x / y, y / x)
        return score
