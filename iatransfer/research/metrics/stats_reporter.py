import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


class StatsReporter:
    def __init__(self, metrics, path):
        self.metric_values = {}
        for metric in metrics.keys():
            self.metric_values[f'{metric}_train'] = []
            self.metric_values[f'{metric}_val'] = []
        self.path = path
        os.makedirs(path, exist_ok=True)

    def update(self, metric_values, is_training=False):
        suf = '_train' if is_training else '_val'
        for metric, value in metric_values.items():
            name = f'{metric}{suf}'
            print(name, value)
            self.metric_values[name].append(value)
            plt.clf()
            for plot_name in [f'{metric}_train', f'{metric}_val']:
                n = len(self.metric_values[plot_name])
                plt.plot(np.arange(n), self.metric_values[plot_name], label=plot_name)
                plt.legend()
            plt.savefig(f'{self.path}/{metric}')
            with open(f'{self.path}/stats.pickle', 'wb') as f:
                pickle.dump(self.metric_values, f)
