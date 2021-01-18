from abc import ABC
from typing import Any, Dict, Tuple, Union

import torch.nn as nn
from inflection import camelize

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.base_transfer import Transfer
from iatransfer.toolkit.transfer.transfer_stats import TransferStats


def _get_subclasses(clazz: type):
    subclasses = {}
    classes = [clazz]
    while len(classes) > 0:
        clazz = classes.pop()
        for c in clazz.__subclasses__():
            subclasses[c.__name__] = c
            classes.append(c)
    return subclasses


class IAT(ABC):
    _standardization_classes = _get_subclasses(Standardization)
    _matching_classes = _get_subclasses(Matching)
    _transfer_classes = _get_subclasses(Transfer)
    standardization = None
    matching = None
    transfer = None

    def __init__(self,
                 standardization: Union[Standardization, str, Tuple[Standardization, Dict], Tuple[str, Dict]]
                 = 'blocks',
                 matching: Union[Matching, str, Tuple[Matching, Dict], Tuple[str, Dict]] = 'dp',
                 transfer: Union[Transfer, str, Tuple[Transfer, Dict], Tuple[str, Dict]] = 'clip',
                 *args,
                 **kwargs) -> None:
        self._try_setting('standardization', standardization, Standardization)
        self._try_setting('matching', matching, Matching)
        self._try_setting('transfer', transfer, Transfer)

    def _try_setting(self, key: str, value: Any, clazz: Any) -> None:
        kwargs = {}
        if isinstance(value, tuple):
            value, kwargs = value[0], value[1]
        if isinstance(value, str):
            setattr(self, key, self._find_subclass(key, value)(**kwargs))
        elif isinstance(value, clazz):
            setattr(self, key, value)
        else:
            raise TypeError()

    def _find_subclass(self, key: str, value: str) -> Any:
        classes = getattr(self, f'_{key}_classes')
        trials = [lambda x: x[1],
                  lambda x: camelize(f'{x[1]}_{x[0]}'),
                  lambda x: camelize(f'{x[1].upper()}{camelize(x[0])}')
                  ]
        for trial in trials:
            res = classes.get(trial((key, value)))
            if res is not None:
                return res
        raise ValueError()

    def run(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> TransferStats:
        from_paths, to_paths = self.standardization(from_module, to_module)
        matched_tensors = self.matching(from_paths, to_paths)
        return self.transfer(matched_tensors)

    def __call__(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> TransferStats:
        return self.transfer(from_module, to_module, *args, **kwargs)
