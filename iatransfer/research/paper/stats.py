from typing import Dict, List

import argparse

from iatransfer.utils.file_utils import read_json
from iatransfer.research.paper.plots import draw_epochs_plot
from iatransfer.research.paper.table import create_table
from iatransfer.research.transfer.utils import get_transfer_method_name

PLOT = 'plot'
TABLE = 'table'

def plot(models: List[Dict], methods: List[Dict], path: str, **kwargs) -> None:
    for model in models:
        for method in methods:
            draw_epochs_plot(model, get_transfer_method_name(method), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=[PLOT, TABLE])
    parser.add_argument('-t', '--teacher-models', 
        help="Path to configuration of teacher models")
    parser.add_argument('-s', '--student-models', required=True,
        help="Path to configuration of student models for transfer")
    parser.add_argument('-i', '--ia-methods', required=True,
        help="Path to iatransfer methods configuration")
    parser.add_argument('-p', '--path',
        help="Path to data")
    args = parser.parse_args()

    if args.path is not None:
        path = args.path
    else:
        path = "./stats"
    if args.teacher_models is not None:
        teachers = read_json(args.teacher_models)["models"]
    else:
        teachers = None
    models = read_json(args.student_models)["models"]
    methods = read_json(args.ia_methods)["methods"]
    {TABLE: create_table, PLOT: plot}[args.mode](models, methods, path, teacher_models = teachers)
        