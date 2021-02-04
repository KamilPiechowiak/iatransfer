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
* simple
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
* parametrization
```python
from iatransfer.toolkit import IAT


iat = IAT(standardization='blocks', matching='dp', score='autoencoder', transfer='trace')

# ==== or

iat = IAT(matching=('dp', {'param': 'value'}))

# ==== or

from iatransfer.toolkit.matching.dp_matching import DPMatching

iat = IAT(matching=DPMatching())
```
* plugins
```python
from iatransfer.toolkit.base_matching import Matching


class CustomMatching(Matching):

    def match(self, from_module, to_module, *args, **kwargs)
        # provide your implementation


# This will instantiate the above CustomMatching in IAT
iat = IAT(matching='custom') 
```
## Citation
When referring to or using iatransfer in a scientific publication, please consider including citation to the following thesis:<br /><br />
@manual{<br />
&emsp;iat2021,<br />
&emsp;title        = {Inter-Architecture Knowledge Transfer},<br />
&emsp;author       = {Maciej A. Czyzewski and Daniel Nowak and Kamil Piechowiak},<br />
&emsp;note         = {Transfer learning between different architectures},organization = {Poznan University of Technology},<br />
&emsp;type = {Bachelor’s Thesis},<br />
&emsp;address = {Poznan, Poland},<br />
&emsp;year = {2021}<br />
}<br /><br />

## Development

#### Init:
```bash
./dev/init.sh
```

#### Run tests:
```bash
nosetests tests
```
#### Install in edit mode:
```
pip install -e .
```

## Research reproduction:
Copy the source code to the GCP cloudshell or install `iatransfer_research` package.

Run:
```bash
/bin/bash ./scripts/research/iatransfer_full_run.sh
```
or
```bash
iatransfer_full_run.sh
```
if `iatransfer_research` has been installed.

