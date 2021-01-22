import argparse
import os
import random
from typing import List

import numpy as np
import torch

from iatransfer.research.train.pretrain_models import train_models
from iatransfer.research.transfer.eval_transfer import eval_transfer
from iatransfer.utils.file_utils import read_json

os.environ['XLA_USE_BF16'] = '1'  # use bfloat16 on tpu

RANDOM_STATE = 13
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PRETRAIN = 'pretrain'
TRANSFER = 'transfer'


def pretrain(args: List[str]) -> None:
    FLAGS = read_json(args.flags)
    models = read_json(args.models)["models"]
    train_models(models, FLAGS)


def transfer(args: List[str]) -> None:
    FLAGS = read_json(args.flags)
    teachers = read_json(args.teacher_models)["models"]
    students = read_json(args.student_models)["models"]
    kwargs = {}
    if args.ia_methods is not None:
        kwargs["transfer_methods"] = read_json(args.ia_methods)["methods"]
    else:
        kwargs["transfer_methods"] = {"transfer": "ClipTransfer"}
    eval_transfer(teachers, students, FLAGS, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=[PRETRAIN, TRANSFER])
    parser.add_argument(
        '-m', '--models', help="Path to configuration of models for pretraining")
    parser.add_argument('-t', '--teacher-models',
                        help="Path to configuration of teacher models for transfer")
    parser.add_argument('-s', '--student-models',
                        help="Path to configuration of student models for transfer")
    parser.add_argument(
        '-f', '--flags', help="Path to trainig flags", required=True)
    parser.add_argument('-i', '--ia-methods',
                        help="Path to iatransfer methods configuration")
    args = parser.parse_args()
    if args.mode == PRETRAIN:
        if args.models is None:
            parser.error("pretrain mode requires --models flag")
        pretrain(args)
    elif args.mode == TRANSFER:
        if args.teacher_models is None or args.student_models is None:
            parser.error(
                "transfer option requires --teacher-models and --student-models flags")
        transfer(args)
