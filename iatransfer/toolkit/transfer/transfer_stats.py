from typing import NamedTuple


class TransferStats(NamedTuple):
    matched_from: int
    matched_to: int
    left_from: int
    left_to: int
    all_from: int
    all_to: int
