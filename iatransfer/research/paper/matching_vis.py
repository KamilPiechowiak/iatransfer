from typing import Dict, List, Union, Tuple

from torch import nn
from pprint import pprint

from iatransfer.toolkit.standardization.graph_standardization import GraphStandardization
from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.tli import get_graph, show_remap, transfer
from iatransfer.research.models.cifar10_resnet import Cifar10Resnet

def get_model_graph_and_ids_mapping(model: nn.Module) -> ['Graph', Dict[nn.Module, int]]:
    graph = get_graph(model)
    names_to_layers_mapping = {}
    def dfs(model: nn.Module, name_prefix: List[str]):
        for child_name, child in model.named_children():
            dfs(child, name_prefix + [child_name])
        names_to_layers_mapping[".".join(name_prefix)] = model
    dfs(model, [])
    
    layers_to_ids_mapping = {}
    for node in graph.nodes.values():
        if node.name.endswith(".weight"):
            layer = names_to_layers_mapping[node.name.replace(".weight", "")]
            layers_to_ids_mapping[layer] = (node.idx, layers_to_ids_mapping.get(layer, (None, None))[1])
        elif node.name.endswith(".bias"):
            layer = names_to_layers_mapping[node.name.replace(".bias", "")]
            layers_to_ids_mapping[layer] = (layers_to_ids_mapping.get(layer, (None, None))[0], node.idx)
    
    return graph, layers_to_ids_mapping

def visualize_matching(matching: Matching, teacher: nn.Module, student: nn.Module):
    teacher_graph, teacher_mapping = get_model_graph_and_ids_mapping(teacher)
    student_graph, student_mapping = get_model_graph_and_ids_mapping(student)
    m = matching.match(teacher, student)
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
    import timm
    m = DPMatching()
    # a = Cifar10Resnet(2)
    # b = Cifar10Resnet(3, no_channels=10)
    a = timm.create_model("mixnet_s")
    b = timm.create_model("mixnet_m")
    # visualize_matching(m, a, b)
    pprint([(k, n.__dict__) for k,n in get_graph(a).nodes.items() if isinstance(n, object)])
    # transfer(a, b)
    # visualize_matching(m, a, b)
