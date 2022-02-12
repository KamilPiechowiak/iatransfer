#!/bin/bash

MACHINE_TYPE=e2-standard-2
ZONE=us-central1-f
VM=vm

gcloud compute instances create $VM \
--image-family=torch-xla \
--image-project=ml-images \
--zone $ZONE \
--subnet default \
--machine-type $MACHINE_TYPE \
--quiet

zip -r weights-transfer.zip iatransfer config scripts -x *__pycache__*
gcloud compute scp weights-transfer.zip $VM:. --zone=$ZONE

gcloud compute ssh $VM --zone=$ZONE

#vm begin
unzip -oq weights-transfer.zip
gcloud auth login
mkdir res
mkdir data
conda activate torch-xla-1.10
# sudo apt-get install libjpeg-dev zlib1g-dev
pip3 install timm tqdm matplotlib inflection networkx


TPU_INSTANCE_NAME=tpu
ZONE=us-central1-f

gcloud compute tpus create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.10  \
--accelerator-type=v2-8

TPU_FULL_ADDRESS=$(gcloud compute tpus list --zone $ZONE | tail -n 1 | tr -s ' ' | cut -d\ -f5 | cut -d/ -f1)
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_FULL_ADDRESS:8470"

python3 -m iatransfer.research.runner pretrain -m config/models_new/cifar10.json