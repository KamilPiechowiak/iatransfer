from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_transfer import Transfer
from iatransfer.toolkit.matching.dp_matching import DPMatching


class MatchedTransfer(Transfer):

    def __init__(self, matching_strategy: Matching = DPMatching(), **kwargs) -> None:
        self.matching_strategy = matching_strategy
        super().__init__(**kwargs)
