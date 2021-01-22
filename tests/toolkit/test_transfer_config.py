import unittest
from typing import List, Dict

import timm

from iatransfer.research.transfer.utils import get_transfer_method_name
from iatransfer.toolkit import IAT
from iatransfer.utils.file_utils import read_json


class TransferConfigTest(unittest.TestCase):

    def transfer(self, models: List[Dict], methods: List[Dict]) -> None:
        for student_json in models:
            if student_json["model"]["supplier"] != "timm.create_model":
                continue
            student = timm.create_model(*student_json["model"]["args"])
            for teacher_name in student_json["teachers"]:
                teacher = timm.create_model(teacher_name.replace("-", "_"))
                for method_json in methods:
                    method = get_transfer_method_name(method_json)
                    print(f"{method}: {teacher_name} -> {student_json['model']['name']}")
                    iat = IAT(**method_json)
                    iat(teacher, student)
    
    def test_transfer_methods(self) -> None:
        models = read_json("config/transfer/methods/models-small.json")["models"]
        methods = read_json("config/transfer/methods/methods-append.json")["methods"]
        self.transfer(models, methods)