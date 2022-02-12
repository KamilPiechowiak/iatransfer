import operator


class ModelFreezer:
    classifier_path = {
        "tf_efficientnet_b0": "classifier",
        "tf_efficientnet_b1": "classifier",
        "tf_efficientnet_b2": "classifier",
        "mixnet_l": "classifier",
        "mixnet_m": "classifier",
        "rexnet_150": "head.fc",
        "rexnet_100": "head.fc",
        "mnasnet_100": "classifier",
        "semnasnet_100": "classifier"
    }

    def freeze(self, model, name):
        for param in model.parameters():
            param.requires_grad = False
        classifier = operator.attrgetter(self.classifier_path[name])(model)
        classifier.weight.requires_grad = True
        classifier.bias.requires_grad = True

    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True
