import csv
import numpy as np
import timm
import torchvision

from iatransfer.toolkit.matching.dp_matching import DPMatching
from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization
from iatransfer.toolkit.standardization.graph_standardization import GraphStandardization
from iatransfer.toolkit.score.shape_score import ShapeScore


def generate_similarity_matrix(path, matcher):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        models_names = [line[0] for line in reader]
    models_names+=["efficientnet_b0"]
    models_names.sort()
    # models_names = models_names[:5]
    print(models_names)
    n = len(models_names)
    sim = np.zeros((n, n))
    models = [timm.create_model(model_name) for model_name in models_names]
    b = GraphStandardization()
    matcher.set_score(ShapeScore())
    for i in range(n):
        for j in range(n):
            sim[i, j] = matcher.sim(
                b.standardize(models[j]),
                b.standardize(models[i])
            )
    print(sim)

def compare_model_to_others(path, model_name, matcher):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        models_names = [line[0] for line in reader]
    model = timm.create_model(model_name)
    models = [timm.create_model(model_name) for model_name in models_names]
    n = len(models)
    sim = np.zeros(n)
    b = BlocksStandardization()
    matcher.set_score(ShapeScore())
    for i in range(n):
        sim[i] = matcher.sim(b.standardize(models[i]), b.standardize(model))
    res = list(zip(models_names, sim))
    res.sort(key=lambda x: x[1])
    print(res)

if __name__ == "__main__":
    generate_similarity_matrix("config/sim/models_subset.csv", DPMatching())
    # generate_similarity_matrix("config/sim/models_subset.csv", DPMatching(GraphStandardization()))
    # compare_model_to_others("config/sim/models.csv", "efficientnet_b0", DPMatching())
    # compare_model_to_others("config/models.csv", "efficientnet_b0", DPMatching(GraphStandardization()))