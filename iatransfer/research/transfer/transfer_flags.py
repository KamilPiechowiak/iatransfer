from iatransfer.research.train.pretrain_flags import FLAGS

FLAGS['epochs'] = 4
FLAGS['repeat'] = 1
FLAGS['learning_rate'] = 1e-3
FLAGS['bucket_path'] = 'gs://weights-transfer/transfer-greatest-weights'

# change flags for transfer
