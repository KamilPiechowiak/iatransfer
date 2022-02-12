#!/bin/bash

python3 -m iatransfer.research.paper.stats plot -s config/transfer/efficientnets-cifar100.json -i config/transfer/method.json
python3 -m iatransfer.research.paper.stats plot -s config/transfer/efficientnets.json -i config/transfer/method.json
python3 -m iatransfer.research.paper.stats plot -s config/transfer/epochs-cifar100.json -i config/transfer/method.json
python3 -m iatransfer.research.paper.stats plot -s config/transfer/small.json -i config/transfer/method.json