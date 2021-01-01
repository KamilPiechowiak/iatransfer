from typing import NamedTuple, List

from torch import nn


class TransferStats(NamedTuple):
    pass


def get_absmeans(module: nn.Module) -> List[float]:
    return [layer.float().abs().mean() for layer in module.state_dict().values()]
