import pickle
from typing import Dict, Callable, Tuple, List

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch import nn

from iatransfer.research.data.data import TrainingTuple, get_dataset, get_dataset_name
from iatransfer.research.models.models import training_tuples
from iatransfer.research.train.train_model import train_model
from iatransfer.research.transfer.transfer_flags import FLAGS


def create_pretrained_models_dict() -> Dict[str, TrainingTuple]:
    pretrained_models: Dict[str, TrainingTuple] = {}
    for t in training_tuples:
        pretrained_models[t.name] = t
    return pretrained_models


SERIAL_EXEC = xmp.MpSerialExecutor()


def test_transfer(transfer_tuples: List[Tuple[TrainingTuple, str]], transfer: Callable[[nn.Module, nn.Module], None]):
    def __mp_fn(rank, transfer_tuples):
        # print(xm.xrt_world_size()) #check number of nodes
        device = xm.xla_device()
        score = 0.0
        for t, from_model_name in transfer_tuples:
            FLAGS['batch_size'] = t.batch_size
            for i in range(FLAGS['repeat']):
                from_path = f'{FLAGS["path"]}/{from_model_name}_{get_dataset_name(t.dataset_tuple)}_{i}'
                from_model: nn.Module = pretrained_models[from_model_name].model()
                from_model.load_state_dict(torch.load(f'{from_path}/best.pt')['model'])
                to_path = f'{FLAGS["path"]}/{t.name}_{get_dataset_name(t.dataset_tuple)}_{i}_from_{from_model_name}'
                to_model = t.model()
                transfer(from_model, to_model)
                train_dataset, val_dataset = get_dataset(t.dataset_tuple)
                train_model(FLAGS, device, to_model, to_path, SERIAL_EXEC.run(train_dataset),
                            SERIAL_EXEC.run(val_dataset))
                if xm.is_master_ordinal():
                    score += get_best_accuracy(to_path) / get_best_accuracy(from_path)
        xm.master_print(f'Score: {score / FLAGS["repeat"] / len(transfer_tuples)}')

    def get_best_accuracy(path):
        with open(f'{path}/stats.pickle', 'rb') as f:
            metric_values = pickle.load(f)
        return max(metric_values['acc_train'])

    pretrained_models = create_pretrained_models_dict()

    xmp.spawn(_mp_fn, args=(transfer_tuples,), nprocs=FLAGS['num_cores'],
              start_method='fork')


def test_transfer_locally(transfer_tuples: List[Tuple[TrainingTuple, str]],
                          transfer: Callable[[nn.Module, nn.Module], None]):
    pretrained_models = create_pretrained_models_dict()
    for t, from_model_name in transfer_tuples:
        from_model: nn.Module = pretrained_models[from_model_name].model()
        to_model = t.model()
        transfer(from_model, to_model)


if __name__ == '__main__':
    from iatransfer.toolkit.transfer import transfer
    from iatransfer.research.models.models import transfer_tuples

    test_transfer(transfer_tuples, transfer)
