import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from iatransfer.research.paper.utils import get_stats_from_file


def draw_epochs_plot(data: Dict[str, dict], method: str, base_path: str, key: str="acc_val"):
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
        plt.plot(np.arange(n) + 1, line[1][:n], label=line[0])
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{model}_{dataset}"
    if "append_to_name" in data:
        plot_path += f"_{data['append_to_name']}"
    plt.legend()
    plt.savefig(plot_path)


def get_module_size(model: nn.Module) -> int:
    size = 0
    for p in model.parameters():
        size += p.numel()
    return size
