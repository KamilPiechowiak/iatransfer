from typing import Any

import json


def read_contents(filename: str, encoding: str = 'utf-8') -> str:
    with open(filename, 'r', encoding=encoding) as file:
        return file.read()


def read_json(filename: str) -> dict:
    with open(filename) as json_file:
        return json.load(json_file)

def save_json(obj: Any, filename: str) -> None:
    with open(filename, "w") as json_file:
        json.dump(obj, json_file)