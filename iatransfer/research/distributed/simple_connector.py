from typing import Callable, List, Dict, Tuple

import torch

from .device_connector import DeviceConnector


class SimpleConnector(DeviceConnector):

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def is_master(self) -> bool:
        return True

    def rendezvous(self, name) -> None:
        pass

    def get_samplers(self, train_dataset, val_dataset) -> Tuple[torch.utils.data.Sampler, torch.utils.data.Sampler]:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset
        )
        val_sampler = torch.utils.data.SequentialSampler(
            val_dataset
        )
        return train_sampler, val_sampler

    def wrap_data_loader(self, data_loader, device) -> torch.utils.data.DataLoader:
        return data_loader

    def optimizer_step(self, opt: torch.optim.Optimizer):
        opt.step()

    def all_avg(self, arr: List):
        for i in range(len(arr)):
            arr[i] = torch.mean(arr[i])

    def print(self, msg, flush=False):
        print(msg, flush=flush)

    def save(self, obj: Dict, path: str):
        torch.save(obj, path)

    def run(self, function: Callable, args: List, nprocs: int):
        function(0, *args)
