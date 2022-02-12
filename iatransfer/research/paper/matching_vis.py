from typing import Dict, List, Union, Tuple

from torch import nn
from pprint import pprint

from iatransfer.toolkit.standardization.graph_standardization import GraphStandardization
from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.base_standardization import Standardization
from iatransfer.toolkit.tli import show_remap, transfer
from iatransfer.toolkit.tli_helpers import get_model_graph_and_ids_mapping
from iatransfer.research.models.cifar10_resnet import Cifar10Resnet

def visualize_matching(matching: Matching, standardization: Standardization, teacher: nn.Module, student: nn.Module):
    teacher_graph, teacher_mapping = get_model_graph_and_ids_mapping(teacher)
    student_graph, student_mapping = get_model_graph_and_ids_mapping(student)
    teacher_standardized = standardization.standardize(teacher)
    student_standardized = standardization.standardize(student)
    m = matching.match(teacher_standardized, student_standardized)
    remap = {}
    def create_remap(m: List[Union[List[Tuple[nn.Module, nn.Module]], Tuple[nn.Module, nn.Module]]]):
        for block in m:
            print(block)
            if isinstance(block, list):
                create_remap(block)
            elif block[0] is not None and block[1] is not None:
                teacher_ids = teacher_mapping[block[0]]
                student_ids = student_mapping[block[1]]
                if teacher_ids[0] is not None and student_ids[0] is not None:
                    remap[student_ids[0]] = teacher_ids[0]
                if teacher_ids[1] is not None and student_ids[1] is not None:
                    remap[student_ids[1]] = teacher_ids[1]
    create_remap(m)
    # pprint(remap)
    show_remap(teacher_graph, student_graph, remap)

if __name__ == "__main__":
    from iatransfer.toolkit.matching.dp_matching import DPMatching
    from iatransfer.toolkit.standardization.blocks_standardization import BlocksStandardization
    from iatransfer.toolkit.score.shape_score import ShapeScore
    import timm
    m = DPMatching()
    m.set_score(ShapeScore())
    s = BlocksStandardization()
    # a = Cifar10Resnet(2)
    # b = Cifar10Resnet(3, no_channels=10)
    a = timm.create_model("regnetx_004")
    b = timm.create_model("regnety_004")
    # visualize_matching(m, a, b)
    # pprint([(k, n.__dict__) for k,n in get_graph(a).nodes.items() if isinstance(n, object)])
    # transfer(a, b)
    visualize_matching(m, s, a, b)
