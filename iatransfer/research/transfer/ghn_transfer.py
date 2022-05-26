import copy
import torch
from torch import nn

class GHNTransfer:

    def __init__(self, name: str):
        import sys
        sys.path.append("ppuda-main")
        from ppuda.ghn.nn import GHN2
        self.ghn = GHN2(name.split("_")[1])

    def transfer(self, model: nn.Module):
        model_copy = copy.deepcopy(model)
        model_copy = self.ghn(model_copy)
        with torch.no_grad():
            predicted_params = {name: param for name, param in model_copy.named_parameters()}
            for name, param in model.named_parameters():
                predicted_param = predicted_params.get(name, None)
                if predicted_param is not None:
                    param[:] = predicted_param