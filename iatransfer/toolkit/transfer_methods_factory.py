from typing import Dict

from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_transfer import Transfer

class TransferMethodsFactory:
    
    def __init__(self):
        self.standardization_classes = self._get_subclasses(Standardization)
        self.matching_classes = self._get_subclasses(Matching)
        self.transfer_classes = self._get_subclasses(Transfer)

    def _get_subclasses(self, clazz: type):
        subclasses = {}
        classes = [clazz]
        while len(classes) > 0:
            clazz = classes.pop()
            for c in clazz.__subclasses__():
                subclasses[c.__name__] = c
                classes.append(c)
        return subclasses

    def get_transfer_method(self, config: Dict):
        matching_kwargs = {}
        if "standardization" in config:
            matching_kwargs["standardization"] = self.standardization_classes[config["standardization"]]()
        transfer_kwargs = {}
        if "matching" in config:
            transfer_kwargs["matching"] = self.matching_classes[config["matching"]](**matching_kwargs)
        return self.transfer_classes[config["transfer"]](**transfer_kwargs)
