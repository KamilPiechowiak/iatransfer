# Inter-Architecture Knowledge Transfer

![Release](https://img.shields.io/badge/toolkit-1.0.0-red)
![Release](https://img.shields.io/badge/research-1.0.0-red)
![Platform](https://img.shields.io/badge/platform-linux--64-blue)
![Python](https://img.shields.io/badge/python-x64%203.8.5-blue)
![Pytorch](https://img.shields.io/badge/torch-1.7.1-blue)

## Usage

```python
from torch import nn
from iatransfer.toolkit import IAT

transfer = IAT()

# run training on Model1()
model_from: nn.Module = Model1()

train(model_from)

# instantiate new model
model_to: nn.Module = Model2() 

# perform the transfer
transfer(model_from, model_to)
```

## Pretraining

To pretrain models run:
`python3 -m iatransfer.research.runner pretrain -m config/models/...`

for example:
`python3 -m iatransfer.research.runner pretrain -m config/models/cifar10_local.json`

The configurations we used are available in the `config/models/` directory. We were running our experiments on TPU. If you want to run them on CPU/GPU, use `config/models_gpu/`.

## Transfer

To evaluate the performance of DPIAT run:
`python3 -m iatransfer.research.runner transfer -t config/transfer/all.json -s config/transfer/...`

for example:
`python3 -m iatransfer.research.runner transfer -t config/transfer/all.json -s config/transfer/local.json`

The configurations we used are available in the `config/transfer/` directory. We were running our experiments on TPU. If you want to run them on CPU/GPU, use `config/transfer_gpu/`.

### Comparison with GHN

To compare our solution with GHN, you need to download the ppuda repository. Go to the main directory of our repository and run:
`git pull git@github.com:facebookresearch/ppuda.git`

In the main direcotry run:
`python3 -m iatransfer.research.runner transfer -t config/transfer/all.json -s config/transfer_gpu/ghn.json`

to start training with GHN initialization.

### Running experiments on TPU

To reproduce our results on TPU, follow these steps:
- set variables
```bash
ZONE=us-central1-f
TPU_INSTANCE_NAME=tpu
BUCKET_NAME=...
```
- create tpu instance
```bash
gcloud alpha compute tpus tpu-vm create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--accelerator-type=v2-8 \
--version=v2-alpha
```
- transfer code and ssh to the instance
```bash
zip -r weights-transfer.zip iatransfer config scripts -x *__pycache__*
gcloud alpha compute tpus tpu-vm scp weights-transfer.zip $TPU_INSTANCE_NAME:. --zone=$ZONE

gcloud alpha compute tpus tpu-vm ssh $TPU_INSTANCE_NAME --zone=$ZONE
```
- prepare the environment
```bash
gcloud auth login
unzip -oq weights-transfer.zip
mkdir res
mkdir data
pip3 install timm tqdm matplotlib inflection networkx
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```
- run experiments with the commands in `pretraining` and `transfer` sections

## Development

#### Init:
`./dev/init.sh`

#### Run tests:
`nosetests tests`

#### Install package:
`pip install .`

##### in edit mode:
`pip install -e .`

# Running cloud package:
Copy source code to the cloudshell.

Run:
`python3 setup_research.py sdist` 
`/bin/bash -x ./scripts/research/iatransfer_full_run.sh`
