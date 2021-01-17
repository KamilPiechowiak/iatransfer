from typing import List

from torch import nn

from iatransfer.toolkit.base_transfer import Transfer


def get_absmeans(module: nn.Module) -> List[float]:
    return [layer.float().abs().mean() for layer in module.state_dict().values()]

def run_transfer(amodel: nn.Module, bmodel: nn.Module, transfer: Transfer):
    stats_before = get_absmeans(bmodel)
    transfer(amodel, bmodel)
    stats_after = get_absmeans(bmodel)
    print('\n'.join(
        [str((x, y)) for x, y in zip(stats_before, stats_after)]
    ))