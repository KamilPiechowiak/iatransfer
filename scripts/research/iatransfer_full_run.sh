#!/bin/bash

#cloudshell
gcloud compute instances create $VM \
--image-family pytorch-1-6-xla \
--image-project deeplearning-platform-release \
--preemptible \
--zone $ZONE \
--subnet default \
--machine-type $MACHINE_TYPE \
--quiet

gcloud compute scp --zone $ZONE dist/iatransfer_research-*.tar.gz $VM:.
gcloud compute scp --zone $ZONE iatransfer.zip $VM:

gcloud compute config-ssh
rm ~/.ssh/config
gcloud compute ssh --zone $ZONE $VM

#vm
gcloud auth login
mkdir res
mkdir data
pip install iatransfer_research-*.tar.gz

#prepare datasets
BUCKET="gs://weights-transfer/data"
cd data
gsutil cp $BUCKET/* .
tar -xf fgvc-aircraft-2013b.tar.gz
tar -xf food-101.tar.gz
cd ..
#end prepare datasets

gcloud compute tpus create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.6  \
--accelerator-type=v2-8 \
--quiet

gcloud compute tpus list --zone $ZONE

TPU_FULL_ADDRESS=$(gcloud compute tpus list --zone $ZONE | tail -n 1 | tr -s ' ' | cut -d\  -f4)

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_FULL_ADDRESS"

iatransfer_pretrain

gcloud compute tpus delete $TPU_INSTANCE_NAME --zone $ZONE --quiet

exit
#cloudshell
gcloud compute instances delete $VM --zone $ZONE --quiet