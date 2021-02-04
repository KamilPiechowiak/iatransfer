from abc import ABC
from typing import Any, Dict, Tuple, Union

import torch.nn as nn
from inflection import camelize

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_score import Score
from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.base_transfer import Transfer
from iatransfer.toolkit.transfer.transfer_stats import TransferStats
from iatransfer.utils.dot_dict import DotDict
from iatransfer.utils.subclass_utils import get_subclasses


class IAT(ABC):
    class _IAT:
        standardization: Standardization = None
        matching: Matching = None
        transfer: Transfer = None
        score: Score = None

        def run(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
                -> TransferStats:
            context = {'from_module': from_module, 'to_module': to_module}

            from_paths, to_paths = self.standardization(from_module), self.standardization(to_module)
            self.score.precompute_scores(from_module, to_module)
            self.matching.set_score(self.score)
            matched_tensors = self.matching(from_paths, to_paths, context=context)
            return self.transfer(matched_tensors, context=context)

        def sim(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
                -> float:
            from_paths, to_paths = self.standardization(from_module), self.standardization(to_module)
            self.score.precompute_scores(from_module, to_module)
            self.matching.set_score(self.score)
            return self.matching.sim(from_paths, to_paths)

        def __call__(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
                -> TransferStats:
            return self.run(from_module, to_module, *args, **kwargs)

    def __new__(cls,
                standardization: Union[Standardization, str, Tuple[Standardization, Dict], Tuple[str, Dict]]
                = 'blocks',
                matching: Union[Matching, str, Tuple[Matching, Dict], Tuple[str, Dict]] = 'dp',
                transfer: Union[Transfer, str, Tuple[Transfer, Dict], Tuple[str, Dict]] = 'clip',
                score: Union[Score, str, Tuple[Score, Dict], Tuple[str, Dict]] = 'ShapeScore',
                *args,
                **kwargs) -> _IAT:
        ctx = DotDict()

        ctx._standardization_classes = get_subclasses(Standardization)
        ctx._matching_classes = get_subclasses(Matching)
        ctx._transfer_classes = get_subclasses(Transfer)
        ctx._score_classes = get_subclasses(Score)

        ctx.algorithm = IAT._IAT()

        cls._try_setting(ctx, 'standardization', standardization, Standardization)
        cls._try_setting(ctx, 'matching', matching, Matching)
        cls._try_setting(ctx, 'transfer', transfer, Transfer)
        cls._try_setting(ctx, 'score', score, Score)

        return ctx.algorithm

    @staticmethod
    def _try_setting(ctx, key: str, value: Any, clazz: Any) -> None:
        kwargs = {}
        if isinstance(value, tuple):
            value, kwargs = value[0], value[1]
        if isinstance(value, str):
            setattr(ctx.algorithm, key, IAT._find_subclass(ctx, key, value)(**kwargs))
        elif isinstance(value, clazz):
            setattr(ctx.algorithm, key, value)
        else:
            raise TypeError()

    @staticmethod
    def _find_subclass(ctx, key: str, value: str) -> Any:
        classes = getattr(ctx, f'_{key}_classes')
        trials = [lambda x: x[1],
                  lambda x: camelize(f'{x[1]}_{x[0]}'),
                  lambda x: camelize(f'{camelize(x[1])}{camelize(x[0])}'),
                  lambda x: camelize(f'{x[1].upper()}{camelize(x[0])}'),
                  lambda x: f'{x[1].split("_")[0].upper()}{camelize("_".join(x[1].split("_")[1:]))}{camelize(x[0])}'
                  ]
        for trial in trials:
            res = classes.get(trial((key, value)))
            if res is not None:
                return res
        raise ValueError()
