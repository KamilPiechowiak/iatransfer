# Inter-Architecture Knowledge Transfer
**builds**
![Toolkit](https://github.com/KamilPiechowiak/weights-transfer/workflows/Toolkit%20build/badge.svg)
![Research](https://github.com/KamilPiechowiak/weights-transfer/workflows/Research%20package%20build/badge.svg)


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