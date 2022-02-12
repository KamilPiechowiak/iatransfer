#!/bin/bash

gsutil -m rsync -x ".*pt" -r gs://kamil-piechowiak-weights-transfer/stats stats/stats/
gsutil -m rsync -x ".*pt" -r gs://kamil-piechowiak-weights-transfer/transfer stats/transfer/