from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iatransfer.research.transfer.utils import get_transfer_method_name
from iatransfer.research.paper.utils import get_stats_from_file

def create_table(models: List[Dict], methods: List[Dict], base_path: str) -> None:
    methods_names = [get_transfer_method_name(method) for method in methods]
    
    transfer_pairs_names = []
    for student in models:
        for teacher in student["teachers"]:
            transfer_pairs_names.append(f"{student['model']['name']}_from_{teacher}")
    
    efficiency = np.zeros((len(methods_names), len(transfer_pairs_names)))
    
    j = 0
    max_epochs = 1
    for student in models:
        org_stats = get_stats_from_file(f"{base_path}/stats/{student['model']['name']}_{student['dataset']['name']}_#/stats.pickle")
        for teacher in student["teachers"]:
            for i, method in enumerate(methods_names):
                transfer_stats = get_stats_from_file(f"{base_path}/transfer-test/{method}_{student['model']['name']}_{student['dataset']['name']}_#_from_{teacher}_best/stats.pickle")
                efficiency[i, j] = np.max(transfer_stats[:max_epochs])/np.max(org_stats[:max_epochs])
            j+=1
    efficiency_copy = efficiency.copy()
    efficiency_copy[efficiency_copy < 1] = 1
    res = pd.DataFrame(
        data=np.concatenate((
            efficiency.mean(axis=1, keepdims=True),
            efficiency_copy.mean(axis=1, keepdims=True),
            efficiency
        ), axis=1),
        index=methods_names,
        columns=["mean"] + ["mean_with_1_as_min"]+ transfer_pairs_names
    )
    res.to_csv("plots/methods.csv")

    for i, method in enumerate(methods_names):
        plt.plot(np.arange(efficiency.shape[1]), efficiency[i], label=method)

    plt.legend()
    plt.savefig("plots/methods.pdf")
