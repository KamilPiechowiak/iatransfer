#!/bin/bash

#cloudshell
gcloud compute config-ssh
rm ~/.ssh/config

gcloud compute scp --zone $ZONE dist/iatransfer_research-*.tar.gz $VM:.

gcloud compute ssh --zone $ZONE vm

#vm
gcloud auth login
mkdir res
mkdir data
pip install iatransfer_research-*.tar.gz

#prepare datasets
BUCKET="gs://weights-transfer/data"
cd data || exit
gsutil cp $BUCKET/* .
tar -xf fgvc-aircraft-2013b.tar.gz
tar -xf food-101.tar.gz
cd ..