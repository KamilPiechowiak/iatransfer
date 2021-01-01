#!/bin/bash

#vm
gcloud compute tpus delete $TPU_INSTANCE_NAME --zone $ZONE --quiet

exit
#cloudshell
gcloud compute instances delete $VM --zone $ZONE --quiet