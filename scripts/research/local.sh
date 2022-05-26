#!/bin/bash

ZONE=us-central1-f
TPU_INSTANCE_NAME=tpu
BUCKET_NAME=...

gcloud alpha compute tpus tpu-vm create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--accelerator-type=v2-8 \
--version=v2-alpha

zip -r weights-transfer.zip iatransfer config scripts -x *__pycache__*
gcloud alpha compute tpus tpu-vm scp weights-transfer.zip $TPU_INSTANCE_NAME:. --zone=$ZONE

gcloud alpha compute tpus tpu-vm ssh $TPU_INSTANCE_NAME --zone=$ZONE

#vm begin
gcloud auth login
unzip -oq weights-transfer.zip
mkdir res
mkdir data

cd data
gsutil cp gs://$BUCKET_NAME/data/food-101.tar.gz .
tar -xf food-101.tar.gz
cd ..
# conda activate torch-xla-1.10
# sudo apt-get install libjpeg-dev zlib1g-dev
pip3 install timm tqdm matplotlib inflection networkx

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

nohup python3 -m iatransfer.research.runner pretrain -m config/models_new/pretrained_cont.json 2>&1 > log.log &
nohup python3 -m iatransfer.research.runner transfer -t config/transfer_new/all.json -s config/transfer_new/methods.json 2>&1 > log.log &
nohup python3 -m iatransfer.research.runner transfer -t config/transfer_new/all.json -s config/transfer_new/ghn.json 2>&1 > log.log &

gcloud alpha compute tpus tpu-vm delete $TPU_INSTANCE_NAME --zone=$ZONE
