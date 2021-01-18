from typing import List, Dict

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from iatransfer.research.data.data import TrainingTuple, get_dataset
from iatransfer.research.train.train_model import train_model

SERIAL_EXEC = xmp.MpSerialExecutor()


def train_models(training_tuples: List[Dict], FLAGS: Dict) -> None:
    # Start training processes
    def _mp_fn(rank: int, training_tuples: List[Dict]) -> None:
        # print(xm.xrt_world_size()) #check number of nodes
        device = xm.xla_device()
        print(device)
        for t in training_tuples:
            t = TrainingTuple.from_json(t)
            FLAGS['batch_size'] = t.batch_size
            if not xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            train_dataset, val_dataset = get_dataset(t.dataset_tuple, FLAGS)
            train_dataset = train_dataset()
            val_dataset = val_dataset()
            if xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            for i in range(FLAGS['repeat']):
                train_model(FLAGS, device, t.model(),
                            f'{FLAGS["path"]}/{t.name}_{t.dataset_tuple.name}_{i}', train_dataset,
                            val_dataset)

    xmp.spawn(_mp_fn, args=(training_tuples,), nprocs=FLAGS['num_cores'],
              start_method='fork')
