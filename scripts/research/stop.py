import os
import time

while True:
    li = os.popen("gsutil ls gs://kamil-piechowiak-weights-transfer/stats").read().split("\n")
    if "gs://kamil-piechowiak-weights-transfer/stats/semnasnet_100_pretrained_CIFAR100_0/" in li:
        print("found")
        time.sleep(60)
        print(os.popen("gcloud alpha compute tpus tpu-vm delete tpu --zone=us-central1-f -q").read())
        break
    print("Waiting")
    time.sleep(60)
