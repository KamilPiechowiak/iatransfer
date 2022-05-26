import copy
from typing import Dict, List
import os

import torch
from torch import nn
import timm

from iatransfer.research.data.data import TrainingTuple, get_dataset
from iatransfer.research.distributed.device_connector import DeviceConnector
from iatransfer.research.train.train_model import train_model
from iatransfer.research.transfer.ghn_transfer import GHNTransfer
from iatransfer.research.transfer.utils import get_transfer_method_name
from iatransfer.toolkit.iat import IAT
from iatransfer.utils.file_utils import read_json


def create_pretrained_models_dict(training_tuples: List[Dict]) -> Dict[str, TrainingTuple]:
    pretrained_models: Dict[str, TrainingTuple] = {}
    for t in training_tuples:
        t = TrainingTuple.from_json(t)
        pretrained_models[f"{t.name}_{t.dataset_tuple.name}"] = t
    return pretrained_models


def eval_transfer(training_tuples: List[Dict], transfer_tuples: List[Dict], FLAGS: Dict, connector: DeviceConnector, transfer_methods: List[Dict]) -> None:
    def _mp_fn(rank: int, transfer_tuples: List[Dict]) -> None:
        device = connector.get_device()
        score = 0.0
        iterations = 0
        for t_json in transfer_tuples:
            config = copy.deepcopy(FLAGS)
            config.update(t_json)
            t = TrainingTuple.from_json(t_json)
            if not connector.is_master():
                connector.rendezvous('download_only_once')
            train_dataset, val_dataset = get_dataset(t.dataset_tuple, config)
            if connector.is_master():
                connector.rendezvous('download_only_once')
            if config.get("checkpoints", None) is None:
                config["checkpoints"] = ["best.pt"]
            for from_model_name in config["teachers"]:
                for transfer_method in transfer_methods:
                    connector.print(transfer_method)
                    transfer = IAT(**transfer_method)
                    for checkpoint_filename in config["checkpoints"]:
                        i_start = config.get('repeat_start', 0)
                        for i in range(i_start, i_start + config['repeat']):
                            to_path = f"{config['path']}/{get_transfer_method_name(transfer_method)}_{t.name}_{t.dataset_tuple.name}_{i}_from_{from_model_name}_{checkpoint_filename.replace('.pt', '')}"
                            to_model = t.model()
                            if from_model_name.split("_")[0] == "ghn":
                                transfer = GHNTransfer(from_model_name)
                                transfer.transfer(to_model)
                            elif from_model_name.split("-")[0] == "timmpretrained":
                                print("timm - pretrained")
                                from_model = timm.create_model(
                                    from_model_name.split("-")[1],
                                    pretrained=True,
                                    **t_json["model"]["kwargs"]    
                                )
                                transfer(from_model, to_model)
                            else:
                                if connector.is_master():
                                    if config.get('gcp', True):
                                        bucket_path = os.path.join(config['source_bucket_path'], f"{from_model_name}_{t.dataset_tuple.name}_{0}", checkpoint_filename)
                                        os.system(f"gsutil cp -r {bucket_path} from_model.pt")
                                    else:
                                        from_path = os.path.join(config['path'], f"{from_model_name}_{t.dataset_tuple.name}_{0}", checkpoint_filename)
                                        os.system(f"cp -r {from_path} from_model.pt")
                                    connector.rendezvous('download_model')
                                else:
                                    connector.rendezvous('download_model')
                                from_model: nn.Module = pretrained_models[
                                    f"{from_model_name}_{t.dataset_tuple.name}"].model()
                                from_model.load_state_dict(
                                    torch.load("from_model.pt", map_location=device)['model'])


                                transfer(from_model, to_model)
                            train_model(config, device, connector, to_model, to_path,
                                        train_dataset, val_dataset, repeat_no=i)

                            # if connector.is_master():
                            #     non_transfer_path = f"{config['path']}/{t.name}_{t.dataset_tuple.name}_{i}"
                            #     dscore = get_best_accuracy(
                            #         to_path) / get_best_accuracy(non_transfer_path)
                            #     print(
                            #         f"{from_model_name} -> {t.name}: {dscore}")
                            #     score += dscore
                            iterations += 1
        # connector.print(f"Score: {score / iterations}")

    def get_best_accuracy(path):
        metric_values = read_json(f'{path}/stats.json')
        return max(metric_values['acc_train'])

    pretrained_models = create_pretrained_models_dict(training_tuples)

    connector.run(_mp_fn, args=(transfer_tuples,), nprocs=FLAGS['num_cores'])
