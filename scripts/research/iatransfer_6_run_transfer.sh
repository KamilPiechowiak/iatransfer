#!/bin/bash

gsutil -m rsync -x ".*checkpoint.*" -r gs://weights-transfer/stats res
pip install .
python setup_research.py install
python iatransfer/research/transfer/eval_transfer.py