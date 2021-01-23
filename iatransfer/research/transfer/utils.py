from typing import Dict

def get_transfer_method_name(transfer_method: Dict) -> str:
    name = [transfer_method["transfer"]]
    if "matching" in transfer_method:
        name.append(transfer_method["matching"])
    if "standardization" in transfer_method:
        name.append(transfer_method["standardization"])
    if "score" in transfer_method:
        name.append(transfer_method["score"])
    return "-".join(name)