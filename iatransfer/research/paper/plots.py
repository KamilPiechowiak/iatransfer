import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import pandas as pd

from iatransfer.research.paper.utils import get_stats_from_file


def draw_epochs_plot(data: Dict[str, dict], method: str, base_path: str, key: str="acc_val"):
    model = data['model']['name']
    dataset = data['dataset']['name']
    lines = [(model, get_stats_from_file(f"{base_path}/stats/{model}_{dataset}_#/stats.pickle", key))]
    n = len(lines[0][1])
    max_acc = lines[0][1].max()
    lines += [("baseline", [max_acc]*n)]
    if data.get('checkpoints', None) is None:
        data['checkpoints'] = ['best.pt']
    i = 0
    for from_model in data['teachers']:
        for checkpoint_filename in data['checkpoints']:
            stats = get_stats_from_file(f"{base_path}/transfer/{method}_{model}_{dataset}_#_from_{from_model}_{checkpoint_filename.replace('.pt', '')}/stats.pickle", key)
            n = min(n, len(stats))
            from_name = from_model
            if 'names' in data:
                from_name = data['names'][i]
            lines += [(from_name, stats)]
            i+=1
    plt.clf()
    for line in lines:
        plt.plot(np.arange(n) + 1, line[1][:n], label=line[0])
    os.makedirs("figures", exist_ok=True)
    plot_path = f"figures/plots/{model}_{dataset}"
    csv_path = f"figures/csv/{model}_{dataset}"
    if "append_to_name" in data:
        plot_path += f"_{data['append_to_name']}"
        csv_path += f"_{data['append_to_name']}"
    plt.legend()
    plt.savefig(plot_path)

    df = pd.DataFrame(
        data=np.stack([l[1][:n] for l in lines]),
        index=[l[0] for l in lines],
        columns=[i+1 for i in range(n)]
    )
    df.to_csv(f"{csv_path}.csv")


def get_module_size(model: nn.Module) -> int:
    size = 0
    for p in model.parameters():
        size += p.numel()
    return size
