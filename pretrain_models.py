import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from train_model import trainModel
from data import TrainingTuple, get_dataset, get_dataset_name
from pretrain_flags import FLAGS
from typing import List

SERIAL_EXEC = xmp.MpSerialExecutor()

def trainModels(trainingTuples: List[TrainingTuple]):

  # Start training processes
  def _mp_fn(rank, trainingTuples):
    # print(xm.xrt_world_size()) #check number of nodes
    device = xm.xla_device()
    print(device)
    for t in trainingTuples:
      FLAGS['batch_size'] = t.batch_size
      if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')
      train_dataset, val_dataset = get_dataset(t.dataset_tuple)
      train_dataset = train_dataset()
      val_dataset = val_dataset()
      if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')
      for i in range(FLAGS['repeat']):
        trainModel(FLAGS, device, t.model(), f"{FLAGS['path']}/{t.name}_{get_dataset_name(t.dataset_tuple)}_{i}", train_dataset, val_dataset)
        
  xmp.spawn(_mp_fn, args=(trainingTuples,), nprocs=FLAGS['num_cores'],
          start_method='fork')

if __name__ == "__main__":
  from models import training_tuples
  trainModels(training_tuples)