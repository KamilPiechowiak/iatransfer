# Inter-Architecture Knowledge Transfer

![Toolkit](https://github.com/KamilPiechowiak/weights-transfer/workflows/Toolkit%20build/badge.svg)
![Research](https://github.com/KamilPiechowiak/weights-transfer/workflows/Research%20build/badge.svg)
---
![Release](https://img.shields.io/badge/toolkit-1.0.0-red)
![Release](https://img.shields.io/badge/research-1.0.0-red)
![Platform](https://img.shields.io/badge/platform-linux--64-blue)
![Python](https://img.shields.io/badge/python-x64%203.8.5-blue)
![Pytorch](https://img.shields.io/badge/torch-1.7.1-blue)


## Modele początkowe

Modele początkowe definujemy są w `models.py` w `training_tuples`. Aby wytrenować modele początkowe, należy uruchomić `pretrain_models.py`.
Parametry treningu zdefiniowane są w `pretrain_flags.py`.

## Modele transferowe

Modele transferowe definujemy w `models.py` w `transfer_tuples`. Aby sprawdzić efektywność transferu, należy uruchomić `test_transfer.py`.
Parametry treningu zdefiniowane są w `transfer_flags.py`.
Przykładowa funkcja transferująca znajduje się w `transfer.py`.

# Development

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
