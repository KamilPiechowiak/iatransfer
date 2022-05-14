import copy
from typing import List, Dict

from iatransfer.research.data.data import TrainingTuple, get_dataset
from iatransfer.research.distributed.device_connector import DeviceConnector
from iatransfer.research.train.train_model import train_model


def train_models(training_tuples: List[Dict], FLAGS: Dict, connector: DeviceConnector) -> None:
    # Start training processes
    def _mp_fn(rank: int, training_tuples: List[Dict]) -> None:
        device = connector.get_device()
        print(device)
        for t in training_tuples:
            config = copy.deepcopy(FLAGS)
            config.update(t)
            t = TrainingTuple.from_json(t)
            if not connector.is_master():
                connector.rendezvous('download_only_once')
            train_dataset, val_dataset = get_dataset(t.dataset_tuple, config)
            if connector.is_master():
                connector.rendezvous('download_only_once')
            i_start = config.get('repeat_start', 0)
            for i in range(i_start, i_start + config['repeat']):
                train_model(config, device, connector, t.model(),
                            f'{config["path"]}/{t.name}_{t.dataset_tuple.name}_{i}', train_dataset,
                            val_dataset, repeat_no=i)

    connector.run(_mp_fn, args=(training_tuples,), nprocs=FLAGS['num_cores'])
