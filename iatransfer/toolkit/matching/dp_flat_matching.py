from typing import Tuple, List, Any, Optional

import numpy as np
import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching


class DPFlatMatching(Matching):
    """Dynamic programming flat matching algorithm for IAT.
    """

    def match(self, from_module: List[nn.Module],
              to_module: List[nn.Module], *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        _, matched, _ = self._match_models(from_module, to_module)
        return matched

    def _compute_score(self, from_module: nn.Module, to_module: nn.Module) -> float:
        def are_all_of_this_class(layers: List[nn.Module], clazz: Any) -> bool:
            return all([isinstance(layer, clazz) for layer in layers])

        score = 0
        layers = [from_module, to_module]
        classes = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.Linear
        ]
        for clazz in classes:
            if are_all_of_this_class(layers, clazz):
                score = 1
                for x, y in zip(from_module.weight.shape, to_module.weight.shape):
                    score *= min(x / y, y / x)
                break

        return score

    def _penalty(self, x):
        return -(x + 1) / 2 if x >= 0 else 0

    def _match_models(self, from_module: List[nn.Module], to_module: List[nn.Module]) -> List[
        Tuple[Optional[nn.Module]]]:
        n = len(to_module)
        m = len(from_module)
        dp = np.zeros((n + 1, m + 1))
        transition = np.zeros((n + 1, m + 1, 2), dtype=int)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score = self._compute_score(from_module[j - 1], to_module[i - 1])
                if dp[i, j - 1] > dp[i - 1, j]:
                    dp[i, j] = dp[i, j - 1]
                    transition[i, j] = [i, j - 1]
                else:
                    dp[i, j] = dp[i - 1, j]
                    transition[i, j] = [i - 1, j]
                for k in range(0, m + 1):  # row
                    value = dp[i - 1, k] + score + self._penalty(k - j)
                    if value > dp[i, j]:
                        dp[i, j] = value
                        transition[i, j] = [i - 1, k]

        matched, matched_indices = [], []
        i, j = n, m
        while i > 0 and j > 0:
            if transition[i, j, 0] == i:
                matched.append((from_module[j - 1], None))
                matched_indices.append((j - 1, None))
            else:
                if dp[tuple(transition[i, j])] != dp[i, j]:
                    matched.append((from_module[j - 1], to_module[i - 1]))
                    matched_indices.append((j - 1, i - 1))
                else:
                    matched.append((None, to_module[i - 1]))
                    matched_indices.append((None, i - 1))
            i, j = transition[i, j]
        while i > 0:
            matched.append((None, to_module[i - 1]))
            matched_indices.append((None, i - 1))
            i -= 1
        while j > 0:
            matched.append((to_module[j - 1], None))
            matched_indices.append((j - 1, None))
            j -= 1

        matched.reverse()
        matched_indices.reverse()
        return dp[n][m], matched, matched_indices
