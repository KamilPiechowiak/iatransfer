#!/bin/bash

BUCKET_NAME=...

gcloud compute disks create tpu-disk \
--size 400GB  \
--zone $ZONE \
--type pd-balanced

gcloud alpha compute tpus tpu-vm attach-disk $TPU_INSTANCE_NAME \
 --zone=$ZONE \
 --disk=tpu-disk

gcloud alpha compute tpus tpu-vm ssh $TPU_INSTANCE_NAME --zone=$ZONE

# sudo lsblk
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p /mnt/disks/persist
sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist
sudo chmod a+w /mnt/disks/persist

gsutil cp gs://$BUCKET_NAME/data/ILSVRC2012_devkit_t12.tar.gz /mnt/disks/persist/
gsutil cp gs://$BUCKET_NAME/data/ILSVRC2012_img_train.tar /mnt/disks/persist/
gsutil cp gs://$BUCKET_NAME/data/ILSVRC2012_img_val.tar /mnt/disks/persist/