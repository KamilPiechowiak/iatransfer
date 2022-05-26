# Inter-Architecture Knowledge Transfer

![Release](https://img.shields.io/badge/toolkit-1.0.0-red)
![Release](https://img.shields.io/badge/research-1.0.0-red)
![Platform](https://img.shields.io/badge/platform-linux--64-blue)
![Python](https://img.shields.io/badge/python-x64%203.8.5-blue)
![Pytorch](https://img.shields.io/badge/torch-1.7.1-blue)

## Usage

```python
import torch
from iatransfer.toolkit import IAT

transfer = IAT()

# run training on Model1()
model_from: nn.Module = Model1()

train(model_from)

# instantiate new model
model_to: nn.Module = Model2() 

# enjoy high-accuracy initialization
transfer(model_from, model_to)
```

## Pretraining

To pretrain models run:
`python3 -m iatransfer.research.runner pretrain -m config/models/...`
for example:
`python3 -m iatransfer.research.runner pretrain -m config/models/cifar10_local.json`
The configurations we used are available in the `config/models/` directory.

## Transfer

To evaluate the performance of DPIAT run:
`python3 -m iatransfer.research.runner transfer -t config/transfer/all.json -s config/transfer/...`
for example:
`python3 -m iatransfer.research.runner transfer -t config/transfer/all.json -s config/transfer/local.json`
The configurations we used are available in the `config/transfer/` directory.

### Comparison with GHN

To compare our solution with GHN, you need to download the ppuda repository. Go to the main directory of our repository and use a command:
`git pull git@github.com:facebookresearch/ppuda.git`

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
