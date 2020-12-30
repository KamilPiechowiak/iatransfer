import os
import random
import torch
import numpy as np

os.environ['XLA_USE_BF16'] = '1' #use bfloat16 on tpu

RANDOM_STATE = 13
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# FLAGS = {
#     'learning_rate': 1e-2,
#     'epochs': 3,
#     'scheduler_mocked_epochs': 100,
#     'persist_state_every': 2,
#     'repeat': 1,
#     # 'path': '/content/gdrive/My Drive/projects/weightsTransfer/xla_test3',
#     'path': 'res',
#     'bucket_path': 'gs://kamil-piechowiak-weights-transfer/test',
#     'num_workers': 4,
#     'datasets_path': 'data'
# }
# FLAGS['num_cores'] = 8

FLAGS = {
    'learning_rate': 1e-2,
    'epochs': 50,
    'scheduler_mocked_epochs': 50,
    'persist_state_every': 5,
    'repeat': 5,
    'path': 'res',
    'bucket_path': 'gs://kamil-piechowiak-weights-transfer/stats',
    'num_workers': 4,
    'datasets_path': 'data',
    'num_cores': 8
}