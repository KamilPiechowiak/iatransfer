import argparse
import random
from typing import List

import numpy as np
import torch

from iatransfer.research.train.pretrain_models import train_models
from iatransfer.research.transfer.eval_transfer import eval_transfer
from iatransfer.utils.file_utils import read_json
from iatransfer.research.distributed.device_connector import DeviceConnector

RANDOM_STATE = 13
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PRETRAIN = 'pretrain'
TRANSFER = 'transfer'


def get_device_connector(FLAGS) -> DeviceConnector:
    if FLAGS['tpu']:
        from iatransfer.research.distributed.xla_connector import XlaConnector
        return XlaConnector()
    else:
        from iatransfer.research.distributed.simple_connector import SimpleConnector
        return SimpleConnector()


def pretrain(args: List[str]) -> None:
    obj = read_json(args.models)
    FLAGS = obj["general"]
    models = obj["models"]
    connector = get_device_connector(FLAGS)
    train_models(models, FLAGS, connector)


def transfer(args: List[str]) -> None:
    obj = read_json(args.student_models)
    FLAGS = obj["general"]
    teachers = read_json(args.teacher_models)["models"]
    students = obj["models"]
    kwargs = {}
    if "methods" in obj:
        kwargs["transfer_methods"] = obj["methods"]
    else:
        kwargs["transfer_methods"] = [{"transfer": "ClipTransfer"}]
    connector = get_device_connector(FLAGS)
    eval_transfer(teachers, students, FLAGS, connector, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=[PRETRAIN, TRANSFER])
    parser.add_argument(
        '-m', '--models', help="Path to configuration of models for pretraining")
    parser.add_argument('-t', '--teacher-models',
                        help="Path to configuration of teacher models for transfer")
    parser.add_argument('-s', '--student-models',
                        help="Path to configuration of student models for transfer")
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
