from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iatransfer.research.transfer.utils import get_transfer_method_name
from iatransfer.research.paper.utils import get_stats_from_file
from iatransfer.research.data.data import TrainingTuple
from iatransfer.toolkit.iat import IAT

def create_pretrained_models_dict(training_tuples: List[Dict]) -> Dict[str, TrainingTuple]:
    pretrained_models: Dict[str, TrainingTuple] = {}
    for t in training_tuples:
        t = TrainingTuple.from_json(t)
        pretrained_models[f"{t.name}_{t.dataset_tuple.name}"] = t
    return pretrained_models

def create_table(models: List[Dict], methods: List[Dict], base_path: str, teacher_models = None) -> None:
    methods_names = [get_transfer_method_name(method) for method in methods]
    
    if teacher_models is not None:
        teacher_models = create_pretrained_models_dict(teacher_models)
        sim_acc_df = pd.DataFrame(columns=['teacher', 'student', 'sim', 'acc_ratio'])

    transfer_pairs_names = []
    for student in models:
        for teacher in student["teachers"]:
            transfer_pairs_names.append(f"{student['model']['name']}_from_{teacher}")
    
    efficiency = np.zeros((len(methods_names), len(transfer_pairs_names)))

    iat = IAT()
    
    j = 0
    max_epochs = 4
    for student in models:
        org_stats = get_stats_from_file(f"{base_path}/stats/{student['model']['name']}_{student['dataset']['name']}_#/stats.pickle")
        for teacher in student["teachers"]:
            for i, method in enumerate(methods_names): 
                transfer_stats = get_stats_from_file(f"{base_path}/transfer-test/{method}_{student['model']['name']}_{student['dataset']['name']}_#_from_{teacher}_best/stats.pickle")
                efficiency[i, j] = np.max(transfer_stats[:max_epochs])/np.max(org_stats[:max_epochs])
            
            if teacher_models is not None:
                student_model = TrainingTuple.from_json(student).model()
                teacher_model = teacher_models[f"{teacher}_{student['dataset']['name']}"].model()
                sim = iat.sim(teacher_model, student_model)
                if not np.isnan(efficiency[0, j]):
                    sim_acc_df = sim_acc_df.append({
                        'teacher': teacher,
                        'student': student['model']['name'],
                        'sim': sim,
                        'acc_ratio': efficiency[0, j]
                    }, ignore_index=True)
            j+=1
            
    efficiency[np.isnan(efficiency)] = 0
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
    res.to_csv("figures/methods.csv")

    for i, method in enumerate(methods_names):
        plt.plot(np.arange(efficiency.shape[1]), efficiency[i], label=method)

    plt.legend()
    plt.savefig("figures/methods.pdf")

    if teacher_models is not None:
        plt.clf()
        plt.scatter(sim_acc_df['sim'], sim_acc_df['acc_ratio'])
        plt.xlabel("similarity")
        plt.ylabel("accuracy ratio")
        # print(sim_x, sim_y)
        # print(np.corrcoef(sim_x, sim_y))
        print(sim_acc_df.corr())
        plt.savefig("figures/sim.pdf")
        sim_acc_df.to_csv("figures/sim.csv")
