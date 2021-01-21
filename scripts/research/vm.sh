gcloud auth login

export ZONE=us-central1-f
export VM=vm
export TPU_INSTANCE_NAME=tpu

mkdir res
mkdir data

TPU_FULL_ADDRESS=$(gcloud compute tpus list --zone $ZONE | tail -n 1 | tr -s ' ' | cut -d\  -f4)
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_FULL_ADDRESS"

unzip -o -q iatransfer.zip
pip install -r requirements.txt
pip install -r requirements_research.txt
gsutil -m rsync -x ".*checkpoint.*" -r gs://weights-transfer/stats res

# python -m iatransfer.research.runner transfer \
#     -t config/models/all.json \
#     -s config/transfer/methods/models-small.json \
#     -f config/transfer/methods/flags.json \
#     -i config/transfer/methods/methods.json

python -m iatransfer.research.runner transfer \
    -t config/models/all.json \
    -s config/transfer/efficientnets.json \
    -f config/transfer/flags-efficientnets.json \
    -i config/transfer/method.json