import os
import time

while True:
    li = os.popen("gsutil ls gs://kamil-piechowiak-weights-transfer/transfer").read().split("\n")
    if "gs://kamil-piechowiak-weights-transfer/transfer/ClipTransfer_mnasnet_100_CIFAR100_0_from_mixnet_m_lr3_pretrained_best/" in li:
        print("found")
        time.sleep(60)
        print(os.popen("gcloud alpha compute tpus tpu-vm delete tpu --zone=us-central1-f -q").read())
        break
    print("Waiting")
    time.sleep(60)
