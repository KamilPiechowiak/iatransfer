from typing import Dict

import os
from subprocess import Popen
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from iatransfer.research.data import data
from iatransfer.research.paper.utils import get_stats_from_file
from iatransfer.research.data.data import TrainingTuple


def draw_epochs_plot(data: Dict[str, str], method: str, base_path: str, key: str="acc_val"):
    model = data['model']['name']
    dataset = data['dataset']['name']
    lines = [(model, get_stats_from_file(f"{base_path}/stats/{model}_{dataset}_#/stats.pickle", key))]
    n = len(lines[0][1])
    if data.get('checkpoints', None) is None:
        data['checkpoints'] = ['best.pt']
    for from_model in data['from_models']:
        for checkpoint_filename in data['checkpoints']:
            stats = get_stats_from_file(f"{base_path}/transfer/{method}_{model}_{dataset}_#_from_{from_model}_{checkpoint_filename.replace('.pt', '')}/stats.pickle", key)
            n = min(n, len(stats))
            lines += [(from_model, stats)]
    plt.clf()
    for line in lines:
        plt.plot(np.arange(n)+1, line[1][:n], label=line[0])
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{model}_{dataset}"
    if "append_to_name" in data:
        plot_path += f"_{data['append_to_name']}"
    plt.legend()
    plt.savefig(plot_path)

def get_module_size(model: nn.Module) -> int:
    size = 0.0
    for p in model.parameters():
        size += p.numel()
    return size

# def create_pretrained_models_dict() -> Dict[str, TrainingTuple]:
#     pretrained_models: Dict[str, TrainingTuple] = {}
#     for t in training_tuples:
#         pretrained_models[f"{t.name}"] = t
#     return pretrained_models

# def draw_sizes_plot(base_path: str, key: str="acc_val"):
#     pretrained_models = create_pretrained_models_dict()
#     results = []
#     for training_tuple, from_models in paper_transfer_tuples:
#         model = training_tuple.name
#         dataset = data.get_dataset_name(training_tuple.dataset_tuple)
#         org_acc = np.max(get_stats_from_file(f"{base_path}/stats/{model}_{dataset}_#/stats.pickle", key))
#         org_size = get_module_size(training_tuple.model())
#         for from_model in from_models:
#             from_size = get_module_size(pretrained_models[from_model].model())
#             transfer_acc = np.max(get_stats_from_file(f"{base_path}/transfer/{model}_{dataset}_#_from_{from_model}/stats.pickle", key))
#             results.append((org_size/from_size, transfer_acc/org_acc))
#     results.sort()
#     results = [x for x in results if np.isnan(x[1]) == False] # FIXME comment out when all stats available
#     results = np.array(results)
#     plt.clf()
#     plt.plot(results[:, 0], results[:, 1])
#     plt.savefig("plots/transfer_acc_by_size")

if __name__ == "__main__":
    create_models_plots('stats')
    # draw_sizes_plot('stats')