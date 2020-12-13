import os
import random
import torch
import numpy as np

os.environ['XLA_USE_BF16'] = '1' #use bfloat16 on tpu

RANDOM_STATE = 13
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FLAGS = {
    'learning_rate': 1e-2,
    'epochs': 2,
    'scheduler_mocked_epochs': 100,
    'batch_size': 32,
    'persist_state_every': 2,
    'repeat': 1,
    'path': '/content/gdrive/My Drive/projects/weightsTransfer/xla_test3',
    'num_workers': 4,
    'datasets_path': '/content/data'
}
FLAGS['num_cores'] = 8 if os.environ.get('TPU_NAME', None) else 1