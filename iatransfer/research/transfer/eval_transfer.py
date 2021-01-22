import pickle
from typing import Dict, List

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch import nn

from iatransfer.research.data.data import TrainingTuple, get_dataset
from iatransfer.research.train.train_model import train_model
from iatransfer.research.transfer.utils import get_transfer_method_name
from iatransfer.toolkit.iat import IAT


def create_pretrained_models_dict(training_tuples: List[Dict]) -> Dict[str, TrainingTuple]:
    pretrained_models: Dict[str, TrainingTuple] = {}
    for t in training_tuples:
        t = TrainingTuple.from_json(t)
        pretrained_models[f"{t.name}_{t.dataset_tuple.name}"] = t
    return pretrained_models


SERIAL_EXEC = xmp.MpSerialExecutor()

def eval_transfer(training_tuples: List[Dict], transfer_tuples: List[Dict], FLAGS: Dict, transfer_methods: List[Dict]) -> None:
    def _mp_fn(rank: int, transfer_tuples: List[Dict]) -> None:
        # print(xm.xrt_world_size()) #check number of nodes
        device = xm.xla_device()
        score = 0.0
        iterations = 0
        for t_json in transfer_tuples:
            t = TrainingTuple.from_json(t_json)
            FLAGS['batch_size'] = t.batch_size
            if not xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            train_dataset, val_dataset = get_dataset(t.dataset_tuple, FLAGS)
            train_dataset = train_dataset()
            val_dataset = val_dataset()
            if xm.is_master_ordinal():
                xm.rendezvous('download_only_once')
            if t_json.get("checkpoints", None) is None:
                t_json["checkpoints"] = ["best.pt"]
            for from_model_name in t_json["teachers"]:
                for transfer_method in transfer_methods:
                    transfer = IAT(**transfer_method)
                    for checkpoint_filename in t_json["checkpoints"]:
                        for i in range(FLAGS['repeat']):
                            from_path = f"{FLAGS['path']}/{from_model_name}_{t.dataset_tuple.name}_{i}"
                            from_model: nn.Module = pretrained_models[
                                f"{from_model_name}_{t.dataset_tuple.name}"].model()
                            from_model.load_state_dict(
                                torch.load(f"{from_path}/{checkpoint_filename}")['model'])
                            
                            to_path = f"{FLAGS['path']}/{get_transfer_method_name(transfer_method)}_{t.name}_{t.dataset_tuple.name}_{i}_from_{from_model_name}_{checkpoint_filename.replace('.pt', '')}"
                            to_model = t.model()

                            transfer(from_model, to_model)
                            train_model(FLAGS, device, to_model, to_path,
                                        train_dataset, val_dataset)

                            if xm.is_master_ordinal():
                                non_transfer_path = f"{FLAGS['path']}/{t.name}_{t.dataset_tuple.name}_{i}"
                                dscore = get_best_accuracy(
                                    to_path) / get_best_accuracy(non_transfer_path)
                                print(
                                    f"{from_model_name} -> {t.name}: {dscore}")
                                score += dscore
                            iterations += 1
        xm.master_print(f"Score: {score / iterations}")

    def get_best_accuracy(path):
        with open(f'{path}/stats.pickle', 'rb') as f:
            metric_values = pickle.load(f)
        return max(metric_values['acc_train'])

    pretrained_models = create_pretrained_models_dict(training_tuples)

    xmp.spawn(_mp_fn, args=(transfer_tuples,), nprocs=FLAGS['num_cores'],
              start_method='fork')
