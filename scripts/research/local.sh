export MACHINE_TYPE=e2-standard-16
export ZONE=us-central1-f
export VM=vm
export TPU_INSTANCE_NAME=tpu

gcloud compute instances create $VM \
--image-family pytorch-1-6-xla \
--image-project deeplearning-platform-release \
--preemptible \
--zone $ZONE \
--subnet default \
--machine-type $MACHINE_TYPE \
--quiet

gcloud compute tpus create $TPU_INSTANCE_NAME \
--zone=$ZONE \
--network=default \
--version=pytorch-1.6  \
--accelerator-type=v2-8 \
--quiet

dev/zip.sh
gcloud compute scp --zone $ZONE iatransfer.zip $VM:
gcloud compute ssh --zone $ZONE $VM
