from typing import Dict

from torch import nn

from iatransfer.toolkit.base_score import Score
from iatransfer.toolkit.tli import score_autoencoder
from iatransfer.toolkit.tli_helpers import get_model_graph_and_ids_mapping


class AutoEncoderScore(Score):
    """Auto-encoder score algorithm for IAT.
    """

    def precompute_scores(self, from_model: nn.Module, to_model: nn.Module, *args, **kwargs) \
            -> float:
        teacher_graph, teacher_mapping = get_model_graph_and_ids_mapping(from_model)
        student_graph, student_mapping = get_model_graph_and_ids_mapping(to_model)
        self.scores, teacher_arr, student_arr = score_autoencoder(teacher_graph, student_graph)
        self.teacher_mapping = self._join_dicts(teacher_mapping, dict([(x, i) for i, x in enumerate(teacher_arr)]))
        self.student_mapping = self._join_dicts(student_mapping, dict([(x, i) for i, x in enumerate(student_arr)]))

    def _join_dicts(self, a: Dict, b: Dict) -> Dict:
        res = {}
        for key, value in a.items():
            res[key] = b[value[0]]
        return res

    def score(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> float:
        try:
            return self.scores[self.teacher_mapping[from_module]][self.student_mapping[to_module]]
        except KeyError:
            return 0
