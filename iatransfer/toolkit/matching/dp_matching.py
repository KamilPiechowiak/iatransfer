from typing import Tuple, List, Any, Union

import numpy as np
import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching


class DPMatching(Matching):
    """Dynamic programming matching algorithm for IAT.
    """

    def match(self, from_module: List[Union[nn.Module, List[nn.Module]]],
              to_module: List[Union[nn.Module, List[nn.Module]]], *args, **kwargs) \
            -> List[Union[Tuple[nn.Module, nn.Module], List[Tuple[nn.Module, nn.Module]]]]:
        _, matched, _ = self._match_models(from_module, to_module)
        return matched

    def sim(self, from_module: List[Union[nn.Module, List[nn.Module]]],
            to_module: List[Union[nn.Module, List[nn.Module]]]) -> float:
        score, _, _ = self._match_models(from_module, to_module)
        return score / self._match_models(to_module, to_module)[0]

    def _compute_score(self, from_module: nn.Module, to_module: nn.Module) -> float:
        def are_all_of_this_class(layers: List[nn.Module], clazz: Any) -> bool:
            return all([isinstance(layer, clazz) for layer in layers])

        score = 0
        layers = [from_module, to_module]
        if are_all_of_this_class(layers, list):
            score, _, _ = self._match_models(from_module, to_module)
        else:
            classes = [
                nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.Linear
            ]
            for clazz in classes:
                if are_all_of_this_class(layers, clazz):
                    score = self.score(from_module, to_module)
                    break

        return score

    def _match_models(self, flat_from_module: Union[nn.Module, List[nn.Module]],
                      flat_to_module: Union[nn.Module, List[nn.Module]]) \
            -> Tuple[float, List[Tuple[nn.Module, nn.Module]], List[Tuple[int, int]]]:
        m = len(flat_from_module)
        n = len(flat_to_module)
        dp = np.zeros((n + 1, m + 1))
        transition = np.zeros((n + 1, m + 1, 2))  #
        scores = np.zeros((n + 1, m + 1))
        # reduction_coeff = 0.7
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                scores[i, j] = self._compute_score(flat_from_module[j - 1], flat_to_module[i - 1])

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if dp[i - 1, j] > dp[i, j - 1]:
                    dp[i, j] = dp[i - 1, j]
                    transition[i, j] = [i - 1, j]
                else:
                    dp[i, j] = dp[i, j - 1]
                    transition[i, j] = [i, j - 1]
                cumulative_sum = 0
                for k in range(i, 0, -1):
                    cumulative_sum += scores[k, j]
                    # score = cumulative_sum*current_reduction+dp[k-1,j-1]
                    score = cumulative_sum / (i - k + 1) ** 0.5 + dp[k - 1, j - 1]
                    # let d be the number of layers matched in flat_to_module to the current layer in flat_from_module
                    # max possible score = d*f(d)
                    # we want d*f(d) to be increasing - adding more matchings should give better score
                    # we want f(d) to be decreasing - adding more matchings should give lower score per layer,
                    # thanks to it we encourage dynamic programming not to choose single layer all the time
                    if score > dp[i, j]:
                        dp[i, j] = score
                        transition[i, j] = [k - 1, j - 1]
                    # current_reduction*=reduction_coeff

        matched = []
        matched_indices = []
        i, j = n, m

        while i > 0:
            if transition[i, j, 1] == j:
                i -= 1
                matched.append((None, flat_to_module[i]))
                matched_indices.append((None, i))
                continue
            t = transition[i, j, 0]
            j -= 1
            from_model_layer_included = False
            while t < i:
                i -= 1
                if scores[i + 1, j + 1] > 0:
                    if isinstance(flat_from_module[j], list) and isinstance(flat_to_module[i], list):
                        _, sub_matched, sub_matched_indices = self._match_models(flat_from_module[j], flat_to_module[i])
                        matched.append(sub_matched)
                        matched_indices.append(sub_matched_indices)
                        matched_indices.append((j, i))
                    else:
                        matched.append((flat_from_module[j], flat_to_module[i]))
                        matched_indices.append((j, i))
                    from_model_layer_included = True
                else:
                    matched.append((None, flat_to_module[i]))
                    matched_indices.append((None, i))
            if not from_model_layer_included:
                matched_indices.append((j, None))
                matched.append((flat_from_module[j], None))
        matched.reverse()
        matched_indices.reverse()
        return dp[n][m], matched, matched_indices
