from typing import Dict

import os
from subprocess import Popen
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from iatransfer.research.data import data
from iatransfer.research.data.data import TrainingTuple


def get_stats_from_file(path: str, key: str):
    i = 0
    stats = []
    while os.path.exists(path.replace('#', str(i))):
        with open(path.replace('#', str(i)), "rb") as f:
            stats += [pickle.load(f)[key]]
        i+=1
    stats = np.array(stats).mean(axis=0)
    return stats

def draw_models_plot(data: Dict[str, str], base_path: str, key: str="acc_val"):
    model = data['to_model']
    dataset = data['dataset']
    lines = [(model, get_stats_from_file(f"{base_path}/stats/{model}_{dataset}_#/stats.pickle", key))]
    n = len(lines[0][1])
    for from_model in data['from_models']:
        stats = get_stats_from_file(f"{base_path}/transfer/{model}_{dataset}_#_from_{from_model}/stats.pickle", key)
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

def create_models_plots(base_path: str):
    plots = [
        {
            'to_model': 'resnet_14', 'dataset': 'CIFAR10', 
            'from_models': ['resnet_20', 'resnet_26', 'resnet_32']
        },
        {
            'to_model': 'resnet_14', 'dataset': 'CIFAR10', 
            'from_models': ['resnet_narrow_14', 'resnet_wide_14', 'resnet_narrow_20', 'resnet_wide_20'],
            'append_to_name': 'width'
        },
        # {
        #     'to_model': 'resnet_20', 'dataset': 'CIFAR10', 
        #     'from_models': ['resnet_14', 'resnet_26', 'resnet_32']
        # },
        # {
        #     'to_model': 'resnet_20', 'dataset': 'CIFAR10', 
        #     'from_models': ['resnet_narrow_14', 'resnet_wide_14', 'resnet_narrow_20', 'resnet_wide_20'],
        #     'append_to_name': 'width'
        # },
        # {
        #     'to_model': 'resnet_32', 'dataset': 'CIFAR10', 
        #     'from_models': ['resnet_14', 'resnet_20', 'resnet_26']
        # },
        # {
        #     'to_model': 'resnet_narrow_14', 'dataset': 'CIFAR10', 
        #     'from_models': ['resnet_14', 'resnet_wide_14']
        # },
        # {
        #     'to_model': 'resnet_wide_14', 'dataset': 'CIFAR10', 
        #     'from_models': ['resnet_14', 'resnet_narrow_14']
        # },
        # {
        #     'to_model': 'resnet_14', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_20', 'resnet_26', 'resnet_32']
        # },
        # {
        #     'to_model': 'resnet_14', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_narrow_14', 'resnet_wide_14', 'resnet_narrow_20', 'resnet_wide_20'],
        #     'append_to_name': 'width'
        # },
        # {
        #     'to_model': 'resnet_20', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_14', 'resnet_26', 'resnet_32']
        # },
        # {
        #     'to_model': 'resnet_20', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_narrow_14', 'resnet_wide_14', 'resnet_narrow_20', 'resnet_wide_20'],
        #     'append_to_name': 'width'
        # },
        # {
        #     'to_model': 'resnet_32', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_14', 'resnet_20', 'resnet_26']
        # },
        # {
        #     'to_model': 'resnet_narrow_14', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_14', 'resnet_wide_14']
        # },
        # {
        #     'to_model': 'resnet_wide_14', 'dataset': 'CIFAR100', 
        #     'from_models': ['resnet_14', 'resnet_narrow_14']
        # },
        {
            'to_model': 'efficientnet-b0', 'dataset': 'CIFAR10', 
            'from_models': ['efficientnet-b1', 'efficientnet-b2']
        },
        {
            'to_model': 'efficientnet-b2', 'dataset': 'CIFAR10', 
            'from_models': ['efficientnet-b0', 'efficientnet-b1']
        },
    ]
    for plot in plots:
        draw_models_plot(plot, base_path)

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