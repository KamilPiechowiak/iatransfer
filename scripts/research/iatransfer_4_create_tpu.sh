#!/bin/bash

#vm
gcloud compute tpus create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.6  \
--accelerator-type=v2-8 \
--quiet

gcloud compute tpus list --zone $ZONE

TPU_FULL_ADDRESS=$(gcloud compute tpus list --zone $ZONE | tail -n 1 | tr -s ' ' | cut -d\  -f4)

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_FULL_ADDRESS"