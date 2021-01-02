#!/bin/bash

gsutil -m rsync -x ".*checkpoint.*" -r gs://weights-transfer/stats res
python setup_research.py install
python iatransfer/research/transfer/test_transfer.py