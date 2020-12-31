from typing import List

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from iatransfer.research.data.data import TrainingTuple, get_dataset, get_dataset_name
from iatransfer.research.train.pretrain_flags import FLAGS
from iatransfer.research.train.train_model import train_model

SERIAL_EXEC = xmp.MpSerialExecutor()


def train_models(training_tuples: List[TrainingTuple]):
    # Start training processes
    def _mp_fn(rank, training_tuples):
        # print(xm.xrt_world_size()) #check number of nodes
        device = xm.xla_device()
        print(device)
        for t in training_tuples:
            FLAGS['batch_size'] = t.batch_size
            if not xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            train_dataset, val_dataset = get_dataset(t.dataset_tuple)
            train_dataset = train_dataset()
            val_dataset = val_dataset()
            if xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            for i in range(FLAGS['repeat']):
                train_model(FLAGS, device, t.model(),
                            f'{FLAGS["path"]}/{t.name}_{get_dataset_name(t.dataset_tuple)}_{i}', train_dataset,
                            val_dataset)

    xmp.spawn(_mp_fn, args=(training_tuples,), nprocs=FLAGS['num_cores'],
              start_method='fork')


def main():
    from iatransfer.research.models.models import training_tuples

    train_models(training_tuples)


if __name__ == '__main__':
    main()
