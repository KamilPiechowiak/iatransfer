from typing import Dict

import numpy as np
import os
import pickle

def get_stats_from_file(path: str, key: str = "acc_val") -> np.array:
    print(path)
    i = 0
    stats = []
    while os.path.exists(path.replace('#', str(i))):
        with open(path.replace('#', str(i)), "rb") as f:
            stats += [pickle.load(f)[key]]
        i+=1
    stats = np.array(stats)
    stats[stats < 0.005] = np.nan
    stats = np.nanmean(stats, axis=0)
    return stats