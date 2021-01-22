from typing import List, Dict

import unittest

from pprint import pprint
import timm

from iatransfer.toolkit.transfer_methods_factory import TransferMethodsFactory
from iatransfer.utils.file_utils import read_json

class TransferConfigTest(unittest.TestCase):

    def transfer(self, models: List[Dict], methods: List[Dict]) -> None:
        t = TransferMethodsFactory()
        for student_json in models:
            if student_json["model"]["supplier"] != "timm.create_model":
                continue
            student = timm.create_model(*student_json["model"]["args"])
            for teacher_name in student_json["teachers"]:
                teacher = timm.create_model(teacher_name.replace("-", "_"))
                for method_json in methods:
                    method = t.get_transfer_method(method_json)
                    print(f"{method}: {teacher_name} -> {student_json['model']['name']}")
                    method(teacher, student)
    
    def test_transfer_methods(self) -> None:
        models = read_json("config/transfer/methods/models-small.json")["models"]
        methods = read_json("config/transfer/methods/methods-append.json")["methods"]
        self.transfer(models, methods)