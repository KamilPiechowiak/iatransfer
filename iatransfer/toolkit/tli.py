import collections
import os
import random
import sys
from copy import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from graphviz import Digraph
from karateclub import FeatherNode, NetMF
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def apply_tli(model, teacher=None):
    model_teacher = str_to_model(teacher)
    transfer(model_teacher, model)
    return model


def get_tli_score(model_from, model_to):
    model_a = str_to_model(model_from)
    model_b = str_to_model(model_to)
    sim, _, _, _ = transfer(model_a, model_b)
    return sim


def get_model_timm(name="dla46x_c"):
    try:
        import timm
    except:
        raise Exception("timm package is not installed! try `pip install timm`")

    model = timm.create_model(name, num_classes=10, in_chans=3, pretrained=True)
    return model


def str_to_model(name):
    if isinstance(name, str):
        print(f"loading `{name}` from pytorch-image-models...")
        model = get_model_timm(name)
    else:
        model = name
    return model


def apply_hard_reset(model):
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return model


def fn_inject(from_tensor, to_tensor):
    from_slices, to_slices = [], []
    for a, b in zip(from_tensor.shape, to_tensor.shape):
        if a < b:
            from_slices.append(slice(0, a))
            to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
        elif a > b:
            from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
            to_slices.append(slice(0, b))
        else:
            from_slices.append(slice(0, a))
            to_slices.append(slice(0, b))
    to_tensor[tuple(to_slices)] = from_tensor[tuple(from_slices)]


def get_networkx(edges, dag=True):
    if dag:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges)
    return G


def show_networkx(graph):
    if isinstance(graph, list):
        graph = get_networkx(edges=graph)
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, with_labels=True, arrows=True)
    plt.show()


def dag_split(edges, token, root=None):
    graph = {}
    for a, b in edges:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    edges_split = []
    visited, queue = set(), collections.deque([root])
    while queue:
        stop = False
        node_root = queue.popleft()
        if node_root not in graph:
            continue
        if node_root == token:
            break
        for node in graph[node_root]:
            if node not in visited:
                if node == token:
                    stop = True
                edges_split.append([node_root, node])
                visited.add(node)
                queue.append(node)
        if stop:
            break

    if not edges_split:
        edges_split.append([token, token])
    return edges_split


def graph_splits(edges, nodes=False):
    G = get_networkx(edges)
    order = list(nx.topological_sort(G))
    if len(order) == 0:
        return {}
    idx_src, idx_dst = order[0], order[-1]
    if not nodes:
        nodes = set()
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
    split_map = {}
    for idx in nodes:
        in_tree = dag_split(edges, idx, root=idx_src)
        out_tree = dag_split(edges, idx, root=idx_dst)
        split_map[idx] = {"in-tree": in_tree, "out-tree": out_tree}
    return split_map


def graph_norm(edges, attr=None):
    normal_id_map = {}
    normal_id_iter = [0]
    rev_mask = {}

    def __for_single(idx):
        if not idx in normal_id_map:
            normal_id_map[idx] = normal_id_iter[0]
            rev_mask[normal_id_iter[0]] = idx
            normal_id_iter[0] += 1

    for a, b in edges:
        __for_single(a)
        __for_single(b)

    norm_edges = []
    for a, b in edges:
        norm_edges.append([normal_id_map[a], normal_id_map[b]])

    norm_attr = []
    if attr:
        for i in range(len(normal_id_map.keys())):
            norm_attr.append(attr[rev_mask[i]])

    return norm_edges, rev_mask, norm_attr


def utils_map_to_mask(split_map):
    mask, graphs = [], []
    for key, split_dict in split_map.items():
        for dict_key in split_dict.keys():
            _g, rev_mask, _ = graph_norm(split_dict[dict_key])
            g = get_networkx(_g, dag=False)
            mask.append([key, dict_key])
            graphs.append(g)
    return mask, graphs


def utils_mask_to_map(mask, X):
    split_map = {}
    for i, (key, dict_key) in enumerate(mask):
        if key not in split_map:
            split_map[key] = {}
        split_map[key][dict_key] = X[i]
    return split_map


def split_flow_level(graph):
    edges = []
    for edge in graph.cluster_links:
        cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
        cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
        edges.append([cluster_idx_1, cluster_idx_2])
    return graph_splits(edges)


def split_cluster_level(graph, cluster_idx):
    edges = graph.cluster_map[cluster_idx].edges
    return graph_splits(edges)


def encode_graph(split_map):
    mask, graphs = utils_map_to_mask(split_map)

    from karateclub import GL2Vec

    model = GL2Vec(dimensions=16)
    print("FIT")
    model.fit(graphs)
    print("EMBEDDING")
    X = model.get_embedding()
    print("-------------------->", X.shape)

    return utils_mask_to_map(mask, X)


class TLIConfig(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


embedding_dim = 5
CONFIG = TLIConfig(
    {

        "node_embedding_attributed": FeatherNode(
            eval_points=4, order=4, svd_iterations=100, reduction_dimensions=32
        ),
        "node_embedding_neighbourhood": NetMF(
            dimensions=embedding_dim
        ),

        "autoencoder": MLPRegressor(
            max_iter=100,
            early_stopping=False,
            activation="relu",
            solver="adam",
            tol=0.0001,

            hidden_layer_sizes=(200, 50, 25,),
            warm_start=True,
            learning_rate_init=0.0005,
            alpha=0.001,
            verbose=True,
        ),
        "test_size": 0.05,
        "samples_per_tensor": 10,
    }
)


def E_nodes(edges, attr=None):
    norm_graph, rev_mask, norm_attr = graph_norm(edges, attr=attr)

    if len(rev_mask) == 0:
        return []

    model = (
        CONFIG.node_embedding_attributed
        if attr
        else CONFIG.node_embedding_neighbourhood
    )

    graph = get_networkx(norm_graph, dag=False)
    if attr:
        model.fit(graph, np.array(norm_attr))
        X = model.get_embedding()
    else:
        model.fit(graph)
        X = model.get_embedding()

    print(f"[E_nodes {X.shape}]", end="")

    encoded_nodes = {}
    for i in range(X.shape[0]):
        encoded_nodes[rev_mask[i]] = X[i]
    return encoded_nodes


def F_architecture(graph, mlb=None, mfa=None):
    edges = []
    cluster_feature = {}
    for cluster_idx, cluster in graph.cluster_map.items():
        cluster_feature[cluster_idx] = [len(cluster.nodes) / (1 + len(cluster.edges))]
    for edge in graph.cluster_links:
        cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
        cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
        edges.append([cluster_idx_1, cluster_idx_2])
    P = E_nodes(edges, attr=cluster_feature)

    S = {}
    for cluster_idx in graph.cluster_map.keys():
        edges = graph.cluster_map[cluster_idx].edges

        if len(edges) > embedding_dim:
            obj = E_nodes(edges)
        else:
            obj = {}
            for idx in graph.cluster_map[cluster_idx].nodes:
                obj[idx] = np.array([0.0] * embedding_dim)
        S.update(obj)

    N = {}
    vec = []
    for idx, node in graph.nodes.items():
        vec.append(__encode(node.name))

    vec = mlb.transform(vec)
    vec = mfa.transform(vec)
    vec_final = []
    for i, (idx, node) in enumerate(
            graph.nodes.items()
    ):
        _shape4 = nn.ConstantPad1d((0, 4 - len(node.size)), 0.0)(
            torch.tensor(node.size)
        )

        shape4 = _shape4.type(torch.FloatTensor) / torch.max(1 + _shape4)
        if shape4[0] > shape4[1]:
            rot = 1
        else:
            rot = 0
        _idx_rev = (graph.max_idx - node.idx) / graph.max_idx
        _idx_rev2 = (node.idx) / graph.max_idx
        _level_rev = (graph.max_level - node.level) / graph.max_level
        _level_rev2 = (node.level) / graph.max_level
        _cluster_rev = (graph.max_idx - node.cluster_idx) / graph.max_idx
        _cluster_rev2 = (node.cluster_idx) / graph.max_idx
        _type = 0 if ".bias" in node.name else 1

        vec_final.append(np.array(
            [rot]
            + shape4.tolist()
            + [(_idx_rev + _cluster_rev + _level_rev) / 3,
               (_idx_rev2 + _cluster_rev2 + _level_rev2) / 3, _type]
        ))

    from sklearn import preprocessing

    _pp = preprocessing.StandardScaler()

    vec_final = _pp.fit_transform(vec_final)

    for i, (idx, node) in enumerate(
            graph.nodes.items()
    ):
        N[idx] = np.array(vec_final[i].tolist() + vec[i].tolist())

    print("(encode_graph ended)")
    return P, S, N


def __q(a, b):
    return np.array(a) + np.array(b)


def __shape_score(s1, s2):
    if len(s1) != len(s2):
        return 0
    score = 1
    for x, y in zip(s1, s2):
        score *= min(x / y, y / x)
    return score


def gen_dataset(graph, P, S, N, EG, prefix=""):
    X, y = [], []

    for idx, node in graph.nodes.items():
        if node.type != "W":
            continue

        cluster_idx = node.cluster_idx

        for _ in range(CONFIG.samples_per_tensor):
            p_src = np.array(P[cluster_idx])
            r = np.random.uniform(low=-0.05, high=0.05, size=p_src.shape)
            p_src += r
            s_src = np.array(S[idx])
            r = np.random.uniform(low=-0.05, high=0.05, size=s_src.shape)
            s_src += r
            q_src = p_src.tolist() + s_src.tolist() + list(N[idx]) + \
                    EG[f"{prefix}_{idx}"]["in-tree"].tolist()
            X.append(__q(q_src, q_src))

            y.append(1 + np.random.uniform(low=-0.05, high=0.05))

        q_src = list(P[cluster_idx]) + list(S[idx]) + list(N[idx]) + \
                EG[f"{prefix}_{idx}"]["in-tree"].tolist()

        X.append(__q(q_src, q_src))
        y.append(1)

        def __get_node(cluster_idx=None, type=None):
            r_idx = None
            if cluster_idx is not None:
                nodes = list(graph.cluster_map[cluster_idx].nodes)
            else:
                nodes = list(graph.nodes.keys())
            for _ in range(len(N)):
                r_idx = random.choice(nodes)
                if graph.nodes[r_idx].type == type or not type:
                    break
            return r_idx

        for _ in range(CONFIG.samples_per_tensor):
            r_idx = __get_node(cluster_idx=cluster_idx, type="W")
            r_cluster_idx = cluster_idx
            if idx == r_idx:
                continue

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx]) + \
                    EG[f"{prefix}_{r_idx}"]["in-tree"].tolist()

            N_bonus = 0
            N_dist = np.linalg.norm(N[idx] - N[r_idx])

            if N_dist <= 1:
                N_bonus = (1 - N_dist) / 4

            X.append(__q(q_src, q_dst))
            y.append(
                N_bonus
                + 0.25
                + 0.5 * __shape_score(graph.nodes[idx].size, graph.nodes[r_idx].size)
            )

        for _ in range(CONFIG.samples_per_tensor):
            r_idx = __get_node(cluster_idx=None, type="W")
            r_cluster_idx = graph.nodes[r_idx].cluster_idx
            if r_cluster_idx == cluster_idx:
                continue
            if idx == r_idx:
                continue

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx]) + \
                    EG[f"{prefix}_{r_idx}"]["in-tree"].tolist()

            N_bonus = 0
            N_dist = np.linalg.norm(N[idx] - N[r_idx])

            if N_dist <= 1:
                N_bonus = (1 - N_dist) / 4

            S_bonus = 0
            S_dist = np.linalg.norm(S[idx] - S[r_idx])

            if S_dist <= 1:
                S_bonus = (1 - S_dist) / 4

            X.append(__q(q_src, q_dst))
            y.append(
                N_bonus / 2
                + S_bonus / 2
                + 0.25 * __shape_score(graph.nodes[idx].size, graph.nodes[r_idx].size)
            )

    print("DATASET", np.array(X).shape)

    return X, y


def __encode(x):
    x = x.replace(".weight", "").replace(".bias", "")
    x = x.replace("blocks", "")
    if "Backward" in x:
        x = ""

    _vec = list(x)

    _lvl = [s for s in _vec if s.isdigit()]
    _lvl = "".join(_lvl)
    _vec = list(set(_vec))
    if _lvl:
        _vec.append(_lvl)

    return _vec


def score_autoencoder(graph_src, graph_dst):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.manifold import Isomap

    mlb = MultiLabelBinarizer()

    vec = []

    for idx, node in graph_dst.nodes.items():
        vec.append(__encode(node.name))

    mlb.fit(vec)
    _l1 = len(graph_dst.nodes.keys())
    _l2 = len(graph_dst.cluster_map.keys())

    mfa = Isomap(n_components=min(_l1 // 2, 30), n_neighbors=min(_l1 // 10, 50), p=3)
    _vec = mlb.transform(vec)
    mfa.fit(_vec)

    P_src, S_src, N_src = F_architecture(graph_src, mlb=mlb, mfa=mfa)
    P_dst, S_dst, N_dst = F_architecture(graph_dst, mlb=mlb, mfa=mfa)

    split_map = {}
    for cluster_idx in graph_src.cluster_map.keys():
        _split_map = split_cluster_level(graph_src, cluster_idx)
        for key in _split_map:
            split_map[f"src_{key}"] = _split_map[key]
    print("(graph_src ended)")
    for cluster_idx in graph_dst.cluster_map.keys():
        _split_map = split_cluster_level(graph_dst, cluster_idx)
        for key in _split_map:
            split_map[f"dst_{key}"] = _split_map[key]
    print("(graph_dst ended)")
    EG = encode_graph(split_map)

    X1, y1 = gen_dataset(graph_src, P_src, S_src, N_src, EG, prefix="src")
    X2, y2 = gen_dataset(graph_dst, P_dst, S_dst, N_dst, EG, prefix="dst")
    X = X1 + X2
    y = y1 + y2

    print("DATASET FULL", np.array(X).shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=42
    )

    model = copy(CONFIG.autoencoder)

    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    loss = mean_squared_error(y_test, y_hat)
    print(f" LOSS --> {loss}")

    def __norm_weights(graph):
        arr, imap, i = [], {}, 0
        for _, (idx, node) in enumerate(graph.nodes.items()):
            if node.type != "W":
                continue
            arr.append(idx)
            imap[idx] = i
            i += 1
        return arr, imap

    src_arr, src_map = __norm_weights(graph_src)
    dst_arr, dst_map = __norm_weights(graph_dst)

    n, m = len(src_arr), len(dst_arr)
    scores = np.zeros((n, m))

    for dst_j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]
        dst_type = node_dst.name.split(".")[-1]

        q_dst = (
                list(P_dst[node_dst.cluster_idx])
                + list(S_dst[idx_dst])
                + list(N_dst[idx_dst])
                + list(EG[f"dst_{idx_dst}"]["in-tree"].tolist())
        )

        q_arr = []
        for src_i, idx_src in enumerate(src_arr):
            node_src = graph_src.nodes[idx_src]
            src_type = node_src.name.split(".")[-1]

            q_src = (
                    list(P_src[node_src.cluster_idx])
                    + list(S_src[idx_src])
                    + list(N_src[idx_src])
                    + list(EG[f"src_{idx_src}"]["in-tree"].tolist())
            )
            q_arr.append(__q(q_src, q_dst))
            scores[src_i, dst_j] = __shape_score(node_dst.size, node_src.size)

            if dst_type != src_type:
                scores[src_i, dst_j] = 0

        y_hat = model.predict(q_arr)
        scores[:, dst_j] *= y_hat

    return scores, src_arr, dst_arr


def transfer(model_src, model_dst=None, teacher=None, inject=True, debug=False):
    if model_src and model_dst:

        print("API: V2")
        pass
    elif not model_dst and teacher:

        print("API: V1")
        model_src, model_dst = teacher, model_src
    else:
        raise Exception("where is teacher?! is this a joke?")

    graph_src = get_graph(model_src)
    graph_dst = get_graph(model_dst)

    if debug:
        show_graph(graph_src, ver=3, path="__tli_src")
        show_graph(graph_dst, ver=3, path="__tli_dst")

    scores, src_arr, dst_arr = score_autoencoder(graph_src, graph_dst)

    remap = {}
    n, m = len(src_arr), len(dst_arr)

    beta = 0.5
    smap = copy(scores)
    for _ in range(n * m):
        i, j = np.unravel_index(smap.argmax(), smap.shape)
        smap[i, :] *= beta

        if dst_arr[j] not in remap:
            smap[:, j] = 0
            remap[dst_arr[j]] = src_arr[i]

    window_size = 0.25
    for _dst_j, idx_dst in enumerate(dst_arr[::-1]):
        dst_j = m - _dst_j - 1
        ith = dst_j / m
        shift = max(int(ith * n - window_size * n), 0)
        i = np.argmax(smap[shift:, dst_j]) + shift
        if idx_dst not in remap:
            remap[idx_dst] = src_arr[i]

    window_size = 1
    for _dst_j, idx_dst in enumerate(dst_arr[::-1]):
        dst_j = m - _dst_j - 1
        ith = dst_j / m
        shift = max(int(ith * n - window_size * n), 0)
        i = np.argmax(smap[shift:, dst_j]) + shift
        if idx_dst not in remap:
            remap[idx_dst] = src_arr[i]

    seen = set()
    all_scores = []
    error_n, error_sum = 0, 0
    print(" " * 45 + "[[src]]" + " " * 30 + "[[dst]]")
    for j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]

        idx_src = remap[idx_dst]
        score = scores[src_arr.index(idx_src), j]
        all_scores.append(score)

        name_src = graph_src.nodes[idx_src].name
        name_dst = node_dst.name
        color_code = "\x1b[1;37;40m"
        if name_src != name_dst:
            error_sum += 1
            color_code = "\x1b[1;31;40m"

        color_end = "\x1b[0m"
        print(
            f"src= {idx_src:3} | dst= {idx_dst:3} | "
            + f"S= {round(score, 2):4} | {color_code}{name_src:30}{color_end} / "
            + f"{name_dst:10}"
        )

        seen.add(idx_src)
        error_n += 1

    sim = max(0, min(1, np.mean(all_scores)))

    print("=== MATCH =================")
    n = len(graph_src.nodes.keys())
    print(f"  SIM --> \x1b[0;34;40m{round(sim, 4)}\x1b[0m")
    print(f" SEEN --> {len(seen):5} / {n:5} | {round(len(seen) / n, 2)}")
    print(f"ERROR --> {error_sum:5} / {error_n:5} | {round(error_sum / error_n, 2)}")
    print("===========================")

    if debug:
        show_remap(graph_src, graph_dst, remap, path="__tli_remap")

    if inject:
        p_src_ref = {}
        for name, param in model_src.named_parameters():
            p_src_ref[name] = param
        p_dst_ref = {}
        for name, param in model_dst.named_parameters():
            p_dst_ref[name] = param

        with torch.no_grad():
            for idx_dst, idx_src in remap.items():
                node_src = graph_src.nodes[idx_src]
                node_dst = graph_dst.nodes[idx_dst]
                p_src = p_src_ref[node_src.name]
                p_dst = p_dst_ref[node_dst.name]
                fn_inject(p_src, p_dst)

    return sim, remap, graph_src, graph_dst


class Node:
    def __init__(self):
        self.idx = 0
        self.var = None
        self.type = None
        self.size = ()
        self.level = 1
        self.cluster_idx = 1


class Graph:
    def __init__(self):
        self.nodes = None
        self.edges = None

        self.cluster_map = None
        self.cluster_links = None

        self.max_level = None
        self.max_idx = None


class Cluster:
    def __init__(self):
        self.cluster_idx = 0
        self.nodes = []
        self.edges = []


def make_graph(var, params=None) -> Graph:
    graph = Graph()
    mod_op = ["AddBackward0", "MulBackward0", "CatBackward"]

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    def __get_type(var):
        node = Node()
        node.var = var
        if hasattr(var, "variable"):
            u = var.variable
            node_name = param_map[id(u)]
            size = list(u.size())
            node.name = node_name
            node.size = size
            node.type = "W"
        else:
            node_name = str(type(var).__name__)
            if node_name in mod_op:
                node.type = "OP"
            else:
                node.type = "F"
            node.name = node_name
        return node

    normal_id_map = {}
    normal_id_iter = [0]

    def __normal_id(var):
        __pointer_idx = id(var)
        if __pointer_idx in normal_id_map:
            return normal_id_map[__pointer_idx]
        else:
            normal_id_map[__pointer_idx] = normal_id_iter[0]
            normal_id_iter[0] += 1
            return normal_id_iter[0] - 1

    def __bfs(graph, degree=2):
        nodes = {}
        edges = {}
        _rev_edges = {}
        _level_map = {}
        _mod_op_map = {}
        visited, queue = set(), collections.deque([graph])
        while queue:
            var = queue.popleft()
            idx_root = __normal_id(var)
            if idx_root not in _level_map:
                _level_map[idx_root] = 1
            if idx_root not in _mod_op_map:
                _mod_op_map[idx_root] = idx_root
            if idx_root not in nodes:
                nodes[idx_root] = __get_type(var)
                nodes[idx_root].cluster_idx = idx_root
                nodes[idx_root].type = "OP"
            if idx_root not in edges:
                edges[idx_root] = []
            if idx_root not in _rev_edges:
                _rev_edges[idx_root] = []
            for _u in var.next_functions:
                u = _u[0]
                idx = __normal_id(u)
                if not u:
                    continue
                edges[idx_root].append(idx)
                if idx not in _rev_edges:
                    _rev_edges[idx] = []
                _rev_edges[idx].append(idx_root)
                if u not in visited:
                    _level_map[idx] = _level_map[idx_root] + 1
                    node = __get_type(u)
                    node.idx = idx
                    if node.type == "OP":
                        _mod_op_map[idx] = idx_root
                    else:
                        _mod_op_map[idx] = _mod_op_map[idx_root]
                    node.level = _level_map[idx]
                    node.cluster_idx = _mod_op_map[idx]
                    nodes[idx] = node

                    visited.add(u)
                    queue.append(u)

        if degree:
            visited, queue = set(), collections.deque([graph])
            for idx_root in _rev_edges:

                if len(_rev_edges[idx_root]) >= degree \
                        and nodes[idx_root].type != "W":
                    nodes[idx_root].type = "OP"
            while queue:
                var = queue.popleft()
                idx_root = __normal_id(var)
                for _u in var.next_functions:
                    u = _u[0]
                    idx = __normal_id(u)
                    if not u:
                        continue
                    if u not in visited:
                        node = nodes[idx]
                        if node.type == "OP":
                            _mod_op_map[idx] = idx_root
                        else:
                            _mod_op_map[idx] = _mod_op_map[idx_root]
                        node.cluster_idx = _mod_op_map[idx]
                        nodes[idx] = node
                        visited.add(u)
                        queue.append(u)
        max_level = 0
        for _, node_level in _level_map.items():
            max_level = max(max_level, node_level)
        return nodes, edges, max_level

    if isinstance(var, tuple):
        raise Exception("Lord Dark Tensor: have not implemented that feature")
        sys.exit(1)
        for v in var:
            __bfs(v.grad_fn)
    else:

        nodes, edges, max_level = __bfs(var.grad_fn)

    graph.nodes = nodes
    graph.edges = edges

    graph.cluster_map, graph.cluster_links = make_clusters(graph)
    if len(graph.cluster_map.keys()) <= 1:
        graph.cluster_links.append([0, 0])

    graph.max_level = max_level
    graph.max_idx = normal_id_iter[0]

    return graph


def make_clusters(graph):
    cluster_map = {}
    cluster_links = []
    for idx, node in graph.nodes.items():
        if node.cluster_idx not in cluster_map:
            cluster_map[node.cluster_idx] = Cluster()
        cluster_map[node.cluster_idx].nodes.append(idx)
    for idx_root, edges in graph.edges.items():
        node_root = graph.nodes[idx_root]
        for idx in edges:
            if graph.nodes[idx].type == "OP":
                cluster_links.append([idx, idx_root])
                continue
            cluster_map[node_root.cluster_idx].edges.append([idx, idx_root])
    return cluster_map, cluster_links


def get_graph(model, input=None):
    graph = None
    input_shape = [input] if input else [(3, 32, 32), (1, 31, 31), (3, 224, 224)]
    for _input_shape in input_shape:
        x = torch.randn(32, *_input_shape)
        try:
            x = x.to(device)
            model = model.to(device)
            graph = make_graph(model(x), params=dict(model.named_parameters()))
            break
        except Exception as err:
            print("ERROR", err)
            continue
    if not graph:
        raise Exception("something really wrong!")
    return graph


def get_idx_to_layers_mapping(model: nn.Module, graph: Graph) -> Dict[int, nn.Module]:
    names_to_layers_mapping = {}

    def dfs(model: nn.Module, name_prefix: List[str]):
        for child_name, child in model.named_children():
            dfs(child, name_prefix + [child_name])
        names_to_layers_mapping[".".join(name_prefix)] = model

    dfs(model, [])

    ids_to_layers_mapping = {}
    for node in graph.nodes.values():
        if node.type == "W":
            node_name = node.name.replace(".weight", "").replace(".bias", "")
            layer = names_to_layers_mapping[node_name]
            ids_to_layers_mapping[node.idx] = layer

    return ids_to_layers_mapping


def make_dot(graph, ver=0, prefix="", rankdir="TB"):
    graph_idx = id(graph)

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",

    )

    graph_attr = dict(
        rank="same",

        rankdir=rankdir,

    )

    print(f"graph_idx={graph_idx}")
    graph_name = f"cluster_{graph_idx}"
    dot = Digraph(name=graph_name, node_attr=node_attr, graph_attr=graph_attr)

    cluster_map, cluster_links = graph.cluster_map, graph.cluster_links

    def __show_graph_nodes():
        for idx, node in graph.nodes.items():
            _header_name = (
                    f"[c = {node.cluster_idx} / "
                    + f"l = {node.level} / "
                    + f"idx = {node.idx}]\n{node.name}"
            )
            if node.type == "OP":
                dot.node(prefix + str(idx), _header_name, fillcolor="green")
            elif node.type == "W":
                dot.node(
                    prefix + str(idx),
                    _header_name + f"\n{node.size}",
                    fillcolor="lightblue",
                )
            else:
                dot.node(prefix + str(idx), _header_name)

    def __show_graph_edges():
        for idx_root, edges in graph.edges.items():
            for idx in edges:
                dot.edge(prefix + str(idx), prefix + str(idx_root), color="black")

    def __show_clusters():
        for cluster_idx, cluster in cluster_map.items():
            with dot.subgraph(name=f"cluster_{graph_idx}_{cluster_idx}") as c:
                c.attr(style="filled", color="lightgrey")
                for edge in cluster.edges:
                    c.edge(prefix + str(edge[0]), prefix + str(edge[1]), color="black")
                c.attr(label=f"cluster {cluster_idx}")
                if rankdir == "LR":
                    c.attr(rotate="90", rankdir="LR")

    if ver == 0:
        __show_graph_nodes()
        __show_graph_edges()

    if ver == 1:
        cluster_seen = set()

        for idx, node in graph.nodes.items():
            if node.type == "OP" and node.cluster_idx not in cluster_seen:
                nodes_in_cluster = len(cluster_map[node.cluster_idx].nodes)
                name = f"{node.cluster_idx} ({nodes_in_cluster})"
                dot.node(prefix + str(node.cluster_idx), name, fillcolor="orange")
                cluster_seen.add(node.cluster_idx)

        for edge in cluster_links:
            cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
            cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
            dot.edge(
                prefix + str(cluster_idx_1),
                prefix + str(cluster_idx_2),
                color="darkgreen",
                penwidth="3",
            )

    if ver == 2:
        __show_clusters()

    if ver == 3:
        __show_graph_nodes()

        for edge in cluster_links:
            dot.edge(
                prefix + str(edge[0]),
                prefix + str(edge[1]),
                color="darkgreen",
                minlen="3",
                penwidth="3",
            )

        __show_clusters()

    resize_dot(dot)
    dot.engine = "dot"
    return dot


def resize_dot(dot, size_per_element=0.15, min_size=12):
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
    return size


def show_graph(model, ver=0, path="__tli_debug", input=None):
    if not isinstance(model, Graph):
        graph = get_graph(model, input=input)
    else:
        graph = model
    dot = make_dot(graph, ver=ver, prefix="this")
    dot.render(filename=path)
    os.system(f"rm {path}")
    print("saved to file")


def show_remap(g1, g2, remap, path="__tli_debug", for_edges=False):
    dot_g1 = make_dot(g1, ver=3, prefix="src", rankdir="TB")
    dot_g2 = make_dot(g2, ver=3, prefix="dst", rankdir="LR")

    graph_attr = dict(rankdir="TB", )
    dot = Digraph(name="root", graph_attr=graph_attr)
    dot_g2.graph_attr.update(rotate="90")

    dot_g2.graph_attr.update(compound="True")
    dot_g1.graph_attr.update(compound="True")
    dot.graph_attr.update(compound="True")
    dot.subgraph(dot_g2)
    dot.subgraph(dot_g1)
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("gist_rainbow")
    if not for_edges:
        arr = g1.cluster_map.keys()
    else:
        arr = range(len(remap.keys()))

    colors = cmap(np.linspace(0, 1, len(arr)))
    colors_map = {}
    for (i, color) in zip(arr, colors):
        colors_map[i] = color

    for i, (idx_dst, idx_src) in enumerate(remap.items()):
        if not for_edges:
            color = colors_map[g1.nodes[idx_src].cluster_idx]
        else:
            color = colors_map[i]
        dot.edge(
            "src" + str(idx_src),
            "dst" + str(idx_dst),
            color=to_hex(color),

            constraint="false",
            penwidth="5",
            weight="5",
        )
    dot.render(filename=path)
    os.system(f"rm {path}")
    print("saved to file")


if __name__ == "__main__":
    if True:
        from research_models import get_model_debug, ResNetUNet

        model_debug_small = get_model_debug(seed=42, channels=3, classes=10)
        model_debug_large = get_model_debug(seed=3, channels=3, classes=10)
        model_unet = ResNetUNet(n_class=6)

        show_graph(model_debug_small, ver=0, path="__tli_figure_1_all")
        show_graph(model_debug_small, ver=3, path="__tli_figure_1_graph")

        transfer(model_debug_small, model_debug_large, debug=True)

        show_graph(model_unet, ver=1, path="__tli_figure_unet")

    if False:
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("mnasnet_100")

    if False:
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("efficientnet_lite0")

    if True:
        model_A = get_model_timm("efficientnet_lite0")
        model_B = get_model_timm("efficientnet_lite1")

    if False:
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("efficientnet_lite1")

    if False:
        model_A = get_model_timm("efficientnet_lite0")
        model_B = get_model_timm("efficientnet_lite0")

    if False:
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mixnet_s")

    if False:
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mixnet_m")

    if False:
        model_A = get_model_timm("mixnet_m")
        model_B = get_model_timm("mixnet_s")

    if False:
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("tf_efficientnet_b0_ap")

    if False:
        model_A = get_model_timm("tf_efficientnet_b0_ap")
        model_B = get_model_timm("mnasnet_100")

    if False:
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mnasnet_100")

    if False:
        model_A = get_model_timm("regnetx_002")
        model_B = get_model_timm("efficientnet_lite0")

    transfer(model_A, model_B, debug=True)
