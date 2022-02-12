import argparse
import random
from typing import List, Dict
import logging

import numpy as np
import torch

from iatransfer.utils.file_utils import read_json
from iatransfer.researchput.train.train import train


def list_tasks(config: Dict) -> List[Dict]:
    tasks = []

    def add_task(model, init, iteration, general, method=None):
        nonlocal tasks
        copied_model = general.copy()
        copied_model.update({
            **model
        })
        copied_model.update({
            "init": init,
            "iteration": iteration,
            "method": method,
        })
        tasks += [copied_model]

    for model in config["models"]:
        inits = model["init"]
        if not isinstance(inits, list):
            inits = [inits]
        for init in inits:
            if init not in ["random", "pretrained"]:
                for method in config["methods"]:
                    for iteration in range(config["general"]["repeat"]):
                        add_task(model, init, iteration, config["general"], method)
            for iteration in range(config["general"]["repeat"]):
                add_task(model, init, iteration, config["general"])

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('nodeid', help="Id of the node")
    parser.add_argument('numnodes', help="Number of the nodes")
    args = parser.parse_args()
    tasks = list_tasks(read_json(args.config))
    node_id, num_nodes = int(args.nodeid), int(args.numnodes)

    logging.basicConfig(level=logging.INFO, format=f"{node_id}: %(asctime)s %(message)s")

    RANDOM_STATE = node_id
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    for i in range(node_id, len(tasks), num_nodes):
        logging.info(f"Running task {i}")
        train(tasks[i])
