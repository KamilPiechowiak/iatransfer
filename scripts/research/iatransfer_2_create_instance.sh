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
