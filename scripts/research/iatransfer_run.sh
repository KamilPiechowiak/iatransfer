#!/bin/bash

#locally
MACHINE_TYPE=e2-highcpu-16

gcloud compute instances create vm \
--image-family pytorch-1-6-xla \
--image-project deeplearning-platform-release \
--preemptible \
--zone us-central1-f \
--subnet default \
--machine-type $MACHINE_TYPE

zip -r weights-transfer.zip .
gcloud compute scp weights-transfer.zip vm:.

gcloud compute config-ssh
rm ~/.ssh/config
gcloud compute ssh vm
#vm begin
unzip weights-transfer.zip
gcloud auth login
mkdir res
mkdir data
pip install efficientnet_pytorch > /dev/null 2>&1

#prepare datasets
BUCKET="gs://weights-transfer/data"
cd data
gsutil cp $BUCKET/* .
tar -xf fgvc-aircraft-2013b.tar.gz
tar -xf food-101.tar.gz
cd ..
#end prepare datasets

TPU_INSTANCE_NAME="tpu"
gcloud compute tpus create $TPU_INSTANCE_NAME \
--zone=us-central1-f \
--network=default \
--version=pytorch-1.6  \
--accelerator-type=v2-8

gcloud compute tpus list --zone=us-central1-f

TPU_IP_ADDRESS=[ip address of TPU] #from the command above
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

iatransfer_pretrain

gcloud compute tpus delete $TPU_INSTANCE_NAME --zone=us-central1-f
#vm end
gcloud compute instances delete vm