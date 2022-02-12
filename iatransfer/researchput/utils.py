import os

from typing import Dict

def get_path(config: Dict):
    path_components = []
    if config.get("method", None) is not None:
        path_components+=[
            get_transfer_method_name(config["method"])
        ]

    path_components+=[
        config["model"]["name"],
        config["dataset"]["name"],
        str(config["iteration"]),
    ]

    if config["init"] != "random":
        path_components+=[
            "from",
            config["init"],
        ]
        if config.get("chekpoint", None) is not None:
            path_components+=[
                config["checkpoint"]
            ]
    if config.get("append_to_name", None) is not None:
        path_components+=[
            config["append_to_name"]
        ]

    
    path = os.path.join(config["path"], "_".join(path_components))
    print(path)

    os.makedirs(path, exist_ok=True)

    return path

def get_transfer_method_name(transfer_method: Dict) -> str:
    keys = ["transfer", "matching", "standardization", "score"]
    name = []
    for key in keys:
        if key in transfer_method:
            name.append(transfer_method[key])
    return "-".join(name)

def get_teacher_model_path(config: Dict):
    path_components = [
        config["path"],
        "_".join([config["init"],
            config["dataset"]["name"],
            str(config["iteration"])])
    ]

    return os.path.join(*path_components)