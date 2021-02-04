from typing import Tuple, List, Union, Dict

import torch.nn as nn

from iatransfer.toolkit.base_matching import Matching
from iatransfer.toolkit.tli import transfer, get_graph, Graph


class GraphNewMatching(Matching):
    """PyTorch execution graph analysis matching algorithm for IAT.
    """

    def match(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> List[Union[Tuple[nn.Module, nn.Module], List[Tuple[nn.Module, nn.Module]]]]:
        from_module = kwargs['context']['from_module']
        to_module = kwargs['context']['to_module']
        _, remap, teacher_graph, student_graph = transfer(from_module, to_module)
        teacher_ids_to_layers_mapping = self.get_ids_to_layers_mapping(from_module, teacher_graph)
        student_ids_to_layers_mapping = self.get_ids_to_layers_mapping(to_module, student_graph)
        matching = []
        total, matched = 0, 0
        classes = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.Linear
        ]
        for student_idx, teacher_idx in remap.items():
            total+=1
            try:
                teacher_layer = teacher_ids_to_layers_mapping[teacher_idx]
                student_layer = student_ids_to_layers_mapping[student_idx]
                for clazz in classes:
                    if isinstance(teacher_layer, clazz) and isinstance(student_layer, clazz):
                        matching.append((teacher_layer, student_layer)) 
                        matched+=1
                        break
            except:
                pass
        print("Matched: ", matched/total)
        return matching
    
    def sim(self, from_module: nn.Module, to_module: nn.Module) -> float:
        return transfer(from_module, to_module)[0]

    def get_ids_to_layers_mapping(self, model: nn.Module, graph: Graph) -> Dict[int, nn.Module]:
        graph = get_graph(model)
        names_to_layers_mapping = {}
        def dfs(model: nn.Module, name_prefix: List[str]):
            for child_name, child in model.named_children():
                dfs(child, name_prefix + [child_name])
            names_to_layers_mapping[".".join(name_prefix)] = model
        dfs(model, [])
        
        ids_to_layers_mapping = {}
        for node in graph.nodes.values():
            if node.name.endswith(".weight"):
                layer = names_to_layers_mapping[node.name.replace(".weight", "")]
                ids_to_layers_mapping[node.idx] = layer
                
        return ids_to_layers_mapping
