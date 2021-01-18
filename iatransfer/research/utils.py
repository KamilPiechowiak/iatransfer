import json


def read_json(path: str):
    with open(path) as file:
        return json.load(file)
