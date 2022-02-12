from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Tuple
import torch


class DeviceConnector(ABC):

    @abstractmethod
    def get_device(self) -> torch.device:
        return

    @abstractmethod
    def is_master(self) -> bool:
        return

    @abstractmethod
    def rendezvous(self, name) -> None:
        return

    @abstractmethod
    def get_samplers(self, train_dataset, val_dataset) -> Tuple[torch.utils.data.Sampler, torch.utils.data.Sampler]:
        return

    @abstractmethod
    def wrap_data_loader(self, data_loader, device) -> torch.utils.data.DataLoader:
        return

    @abstractmethod
    def optimizer_step(self, opt: torch.optim.Optimizer):
        return

    @abstractmethod
    def all_avg(self, arr: List):
        return

    @abstractmethod
    def print(self, msg, flush):
        return

    @abstractmethod
    def save(self, obj: Dict, path: str):
        return

    @abstractmethod
    def run(self, function: Callable, args: List, nprocs: int):
        return
