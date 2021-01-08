from typing import List, Tuple, Dict

def topological_sort(edges: List[Tuple[str, str]]) -> Dict[str, int]:
    def dfs(a: 'Node'):
        nonlocal postorder
        a['vis'] = True
        for bid in a['e']:
            b = nodes[bid]
            if not b['vis']:
                dfs(b)
        a['postorder'] = postorder
        postorder+=1
                
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

edges = [(1, 2), (2, 4), (2, 6), (5, 4), (5, 1), (3, 1), (3, 5)]
print(topological_sort(edges))
# print(topological_sort([
#     (1, 3), (6, 1), (5, 4), (4, 3), (4, 2), (3, 7), (3, 8), (8, 7)
# ]))
from collections import Counter
parents_count = Counter([e[1] for e in edges])
print(parents_count)