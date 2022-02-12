export MACHINE_TYPE=e2-standard-2
export ZONE=us-central1-f
export VM=vm2
export TPU_INSTANCE_NAME=tpu2

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


#########################################

gcloud auth login

export ZONE=us-central1-f
export VM=vm2
export TPU_INSTANCE_NAME=tpu2

mkdir res
mkdir data

TPU_FULL_ADDRESS=$(gcloud compute tpus list --zone $ZONE | grep $TPU_INSTANCE_NAME | tr -s ' ' | cut -d\  -f4)
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_FULL_ADDRESS"

unzip -o -q iatransfer.zip
pip install -r requirements.txt
pip install -r requirements_research.txt
gsutil -m rsync -x ".*checkpoint.*" -r gs://weights-transfer/stats res

