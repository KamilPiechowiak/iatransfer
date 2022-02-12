import csv
import numpy as np
import timm
import torchvision

from iatransfer.toolkit import IAT


def generate_similarity_matrix(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        models_names = [line[0] for line in reader]
    models_names.sort()
    # models_names = models_names[:5]
    print(models_names)
    n = len(models_names)
    sim = np.zeros((n, n))
    models = [timm.create_model(model_name) for model_name in models_names]
    iat = IAT()
    for i in range(n):
        for j in range(n):
            sim[i, j] = iat.sim(models[j], models[i])
    print(sim)

def compare_model_to_others(path, model_name):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        models_names = [line[0] for line in reader]
    model = timm.create_model(model_name)
    models = [timm.create_model(model_name) for model_name in models_names]
    n = len(models)
    sim = np.zeros(n)
    iat = IAT()
    for i in range(n):
        sim[i] = iat.sim(models[i], model)
    res = list(zip(models_names, sim))
    res.sort(key=lambda x: -x[1])
    print(f"model & similarity to {model_name}")
    print("\\hline")
    for name, sim in res:
        print(f"{name} & {round(sim, 2)}\\\\")
        print("\\hline")
    print(res)

if __name__ == "__main__":
    # generate_similarity_matrix("config/sim/models_subset.csv")
    # generate_similarity_matrix("config/sim/models_subset.csv", DPMatching(GraphStandardization()))
    compare_model_to_others("config/sim/models_subset.csv", "efficientnet_b0")
    # compare_model_to_others("config/models.csv", "efficientnet_b0", DPMatching(GraphStandardization()))