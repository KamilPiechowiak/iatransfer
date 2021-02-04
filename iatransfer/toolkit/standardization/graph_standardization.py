from __future__ import annotations

import random
from collections import Counter
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from graphviz import Digraph
from torch.autograd import Variable

from iatransfer.toolkit.base_standardization import Standardization


class GraphStandardization(Standardization):
    """Graph standardization algorithm for IAT.
    """

    def standardize(self, module: nn.Module, *args, **kwargs) \
            -> List[Union[nn.Module, List[nn.Module]]]:
        return self.get_blocks(module)

    def make_dot(self, var, params=None):

        def resize_graph(dot, size_per_element=0.15, min_size=12):
            num_rows = len(dot.body)
            content_size = num_rows * size_per_element
            size = max(min_size, content_size)
            size_str = str(size) + "," + str(size)
            dot.graph_attr.update(size=size_str)
            return size

        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
        )

        idx = random.randint(1, 100000)
        dot = Digraph(
            name=f"cluster_{idx}", node_attr=node_attr, graph_attr=dict(size="12,12")
        )
        seen = set()

        def size_to_str(size):
            return "(" + (", ").join(["%d" % v for v in size]) + ")"

        mod_op = ["AddBackward0", "MulBackward0", "CatBackward"]

        lmap, emap, hmap = {}, {}, {}
        elist, clist = [], []

        def add_nodes(var, root=None, c=0, depth=0, branch=0, global_i=0):
            if var in seen:
                return None

            depth += 1
            global_i += 1

            if c not in lmap:
                lmap[c], emap[c] = [], []

            # FIXME: move to function
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                lmap[c].append(
                    {
                        "branch": branch,
                        "depth": depth,
                        "global_i": global_i,
                        "name": name,
                        "size": u.size(),
                        "type": "param",
                        "id": str(id(var)),
                    }
                )
                hmap[name] = str(id(var))
                dot.node(
                    str(id(var)),
                    f"c={c} branch={branch} depth={depth}\n" + node_name,
                    fillcolor="lightblue",
                )
            else:
                node_name = str(type(var).__name__)
                if node_name in mod_op:
                    depth = 0
                    prev_c = c
                    c = str(id(var))
                    clist.append((c, prev_c))
                    emap[c], lmap[c] = [], []
                    dot.node(str(id(var)), node_name + f" [{c}]", fillcolor="green")
                else:
                    lmap[c].append(
                        {
                            "branch": branch,
                            "depth": depth,
                            "global_i": global_i,
                            "name": node_name,
                            "type": "func",
                            "id": str(id(var)),
                        }
                    )
                    dot.node(str(id(var)), node_name + f" [{str(id(var))}]")
            if root:
                emap[c].append((str(id(var)), root))
            seen.add(var)

            if hasattr(var, "next_functions"):
                for _branch, u in enumerate(var.next_functions):
                    if node_name in mod_op:
                        branch = _branch
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)), color="blue")
                        elist.append((str(id(u[0])), str(id(var))))
                        add_nodes(
                            u[0],
                            root=str(id(var)),
                            c=c,
                            depth=depth,
                            branch=branch,
                            global_i=global_i,
                        )

        if isinstance(var, tuple):
            for v in var:
                add_nodes(v.grad_fn)
        else:
            add_nodes(var.grad_fn)

        for c, edges in emap.items():
            with dot.subgraph(name=f"cluster_{idx}_{c}") as _c:
                _c.attr(color="blue")
                _c.attr(style="filled", color="lightgrey")
                _c.node_attr["style"] = "filled"
                _c.edges(edges)
                _c.attr(label=f"cluster {c}")

        return dot, lmap, hmap, clist, elist

    def get_graph(self, model: nn.Module):
        # FIXME: warning about 'torchviz'
        try:
            x = torch.randn(32, 3, 32, 32)
            g, lmap, hmap, clist, elist = self.make_dot(model(x), params=dict(model.named_parameters()))
        except:
            # FIXME: universal head? (what happens if MNIST?)
            print("ERROR: trying one channel")
            x = torch.randn(32, 1, 31, 31)
            g, lmap, hmap, clist, elist = self.make_dot(model(x), params=dict(model.named_parameters()))
        return g, lmap, hmap, clist, elist

    def get_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        layers_dict = {}

        def dfs(model: nn.Module, name_prefix: List[str]):
            for child_name, child in model.named_children():
                dfs(child, name_prefix + [child_name])
            layers_dict[".".join(name_prefix)] = model

        dfs(model, [])
        return layers_dict

    def sort_graph(self, graph: 'Graph', clusters_edges: List[Tuple[str, str]],
                   nodes_edges: List[Tuple[str, str]]) -> 'Graph':

        def topological_sort(edges: List[Tuple[str, str]]) -> Dict[str, int]:
            def dfs(a: 'Node'):
                nonlocal postorder
                a['vis'] = True
                for bid in a['e']:
                    b = nodes[bid]
                    if not b['vis']:
                        dfs(b)
                a['postorder'] = postorder
                postorder += 1

            nodes = set([e[0] for e in edges]) | set([e[1] for e in edges])
            nodes = dict([
                (node, {'id': node, 'e': [], 'postorder': -1, 'vis': False}) for node in nodes
            ])
            for e in edges:
                nodes[e[0]]['e'].append(e[1])
            postorder = 0
            for node in nodes.values():
                if node['postorder'] == -1:
                    dfs(node)
            ordered_nodes = sorted(list(nodes.values()), key=lambda node: -node['postorder'])
            return dict([(node['id'], i) for i, node in enumerate(ordered_nodes)])

        def count_ancestors(edges: List[Tuple[str, str]]) -> Dict[str, int]:
            return

        def sort_cluster(cluster: List['Node']) -> List['Node']:
            def cmp(node: 'Node') -> int:
                if node['type'] == 'param':
                    a = node_child[node['id']]
                    while parents_count[a] == 1:
                        a = node_child[a]
                    return nodes_order[a]
                else:
                    return nodes_order[node["id"]]

            return sorted(cluster, key=cmp)

        clusters_order = topological_sort(clusters_edges)
        node_child = dict(nodes_edges)
        nodes_order = topological_sort(nodes_edges)
        parents_count = Counter([e[1] for e in nodes_edges])

        new_graph = {}
        for cid in clusters_order.keys():
            cluster = graph[cid]
            cluster = sort_cluster(cluster)
            new_graph[cid] = cluster
        return new_graph

    def get_blocks(self, model: nn.Module) -> List[Union[nn.Module, List[nn.Module]]]:
        _, graph, _, clusters_edges, nodes_edges = self.get_graph(model)
        graph = self.sort_graph(graph, clusters_edges, nodes_edges)
        layers_dict = self.get_layers(model)
        layers = []
        for block in graph.values():
            layers_in_block = []
            for layer in block:
                if layer['name'].endswith('.weight'):
                    name = layer['name'].replace('.weight', '')
                    layers_in_block.append(layers_dict[name])
            layers.append(layers_in_block)
        return layers
