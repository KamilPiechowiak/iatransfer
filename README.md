![Toolkit](https://github.com/KamilPiechowiak/weights-transfer/workflows/Toolkit%20build/badge.svg)
![Research](https://github.com/KamilPiechowiak/weights-transfer/workflows/Research%20build/badge.svg)
![Documentation](https://github.com/KamilPiechowiak/weights-transfer/workflows/Documentation/badge.svg)

![Coverage](https://img.shields.io/badge/coverage-95%25-green)
![Release](https://img.shields.io/badge/toolkit-1.0.0-blue)
![Release](https://img.shields.io/badge/research-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Platform](https://img.shields.io/badge/platform-linux--64-blue)
![Python](https://img.shields.io/badge/python-x64%203.8-blue)
![Pytorch](https://img.shields.io/badge/torch-1.7.1-blue)

# Inter-Architecture Knowledge Transfer
iatransfer is a PyTorch package for transferring pretrained weights between models of different architectures instantaneously.

Drastically speed up your training process using two additional lines of code.


## Installation
```bash
pip install iatransfer
```


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

## Cite
When referring to or using iatransfer in a scientific publication, please consider including citation to the following thesis:&nbsp;&nbsp;
@manual{&nbsp;
  iat2021,&nbsp;
  title        = {Inter-Architecture Knowledge Transfer},&nbsp;
  author       = {Maciej A. Czyzewski and Daniel Nowak and Kamil Piechowiak},&nbsp;
  note         = {Transfer learning between different architectures},organization = {Poznan University of Technology},&nbsp;
  type         = {Bachelor’s Thesis},&nbsp;
  address      = {Poznan, Poland},&nbsp;
  year         = {2021}&nbsp;
}&nbsp;&nbsp;

iatransfer has been tested under python 3.8.

## Development

#### Init:
`./dev/init.sh`

#### Run tests:
`nosetests tests`

#### Install package:
`pip install .`

##### in edit mode:
`pip install -e .`

## Research reproduction:
Copy the source code to the GCP cloudshell or install `iatransfer_research` package.

Run:
`/bin/bash -x ./scripts/research/iatransfer_full_run.sh`
