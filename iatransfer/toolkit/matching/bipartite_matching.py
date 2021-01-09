from typing import Tuple, List, Union

import networkx as nx
import numpy as np
import torch.nn as nn

from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.matching.dp_matching import Matching, DPMatching
from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization


class BipartiteMatching(DPMatching):

    def __init__(self, standardization: Standardization = BlocksStandardization()):
        super().__init__(standardization)

    def match(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> List[Tuple[nn.Module, nn.Module]]:
        flattened_from_module = self.standardization.standardize(from_module)
        flattened_to_module = self.standardization.standardize(to_module)
        _, matched, _ = self._match_bipartite(flattened_from_module, flattened_to_module)
        return matched

    def sim(self, from_module: nn.Module, to_module: nn.Module):
        flattened_from_module = self.standardization.standardize(from_module)
        flattened_to_module = self.standardization.standardize(to_module)
        score, _, _ = self._match_bipartite(flattened_from_module, flattened_to_module)
        return score

    def _match_bipartite(self, flat_from_module: Union[nn.Module, List[nn.Module]],
                         flat_to_module: Union[nn.Module, List[nn.Module]]) \
            -> Tuple[float, List[Tuple[nn.Module, nn.Module]], List[Tuple[int, int]]]:
        m = len(flat_from_module)
        n = len(flat_to_module)
        scores = np.zeros((n, m))

        graph = nx.Graph()

        for i in range(n):
            for j in range(m):
                dist_score = 1 - (abs(i / n - j / m) ** 2)
                scores[i, j] = dist_score * self._compute_score(flat_from_module[j], flat_to_module[i])
                graph.add_edge(f'x{i}', f'{j}', weight=scores[i, j])

        max_scores = np.max(scores, axis=0).sum()

        matching = nx.algorithms.matching.max_weight_matching(graph)
        matching = map(lambda x: x if x[0].startswith('x') else (x[1], x[0]), matching)

        matched_indices = [(self._convert(vertex_a), self._convert(vertex_b)) for vertex_a, vertex_b in matching]
        matched_indices_arr = np.array(matched_indices).transpose()

        final_score = scores[matched_indices_arr[0], matched_indices_arr[1]].sum() / max_scores
        matched = [(flat_from_module[vertex_b], flat_to_module[vertex_a]) for vertex_a, vertex_b in matched_indices]
        return final_score, matched, matched_indices

    @staticmethod
    def _convert(vertex: str) -> int:
        return int(vertex[1:]) if vertex.startswith('x') else int(vertex)
