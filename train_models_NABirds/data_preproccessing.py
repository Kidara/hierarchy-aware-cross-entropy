# key features of the dataset:
# classes_birds.txt holds the index to bird name (this will be the output embedding space as well)
# hierarchy.txt maps index of a certain bird to its ancestor
# nab_class_index.unitsphere.json holds the mapping from the actual label in NABirds to our hierarchy of labels

import pandas as pd
from collections import defaultdict
import json
import os
import networkx as nx
import matplotlib.pyplot as plt

DILUTION = 0.7
HIERARCHY_FILE = 'hierarchy.txt'
HIERARCHY_MATRIX_FILE = 'hierarchical_reach_matrix.csv'
NAB_DIR = ''
N = 1011
hierarchy_matrix = pd.DataFrame(0, index = range(N), columns = range(N), dtype = int)
sibling_matrix = pd.DataFrame(0, index = range(N), columns = range(N), dtype = int)
diluted_encoding = pd.DataFrame(0.0, index = range(N), columns = range(N), dtype = float)

# create dictionary from child to direct parent
child_to_parent = {}
with open(HIERARCHY_FILE) as f:
    for line in f:
        birds = line.strip().split()
        if len(birds) == 2:
            child, parent = int(birds[0]), int(birds[1])
            child_to_parent[child] = parent

def dfs_ancestors(current_node, leaf_node):
    '''
    fills hierarchical reach matrix for the given node
    '''
    if current_node not in child_to_parent:
        return

    parent = child_to_parent[current_node]
    hierarchy_matrix.loc[parent, current_node] = 1
    hierarchy_matrix.loc[parent,leaf_node] = 1
    dfs_ancestors(parent, leaf_node)

def soft_one_hot(leaf, node, probability):
    """
    args: one leaf, one other node, and the probability distribution amount remaining
    returns: none
    fills soft probability distribution matrix with encoding for label
    """
    if node not in child_to_parent:
        diluted_encoding.loc[node, leaf] += probability
        return
    
    diluted_encoding.loc[node, leaf]  += DILUTION * probability
    parent = child_to_parent[node]
    soft_one_hot(leaf, parent, (1 - DILUTION) * probability)
    return

for node in range(0, N):
    hierarchy_matrix.loc[node,node] = 1
    dfs_ancestors(node, node)
    soft_one_hot(node, node, 1)

#sibling_matrix[i,j] = 1 if i and j share the same parent (i can be equal to j)
parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

for parent, children in parent_to_children.items():
    for i in children:
        for j in children:
            sibling_matrix.loc[i, j] = 1

print(f"sum of columns for soft one hot encodings: {diluted_encoding.sum(axis=0)}")
hierarchy_matrix.to_csv('hierarchy_matrix.csv', index=True, header=True)
diluted_encoding.to_csv('diluted_encoding.csv', index=True, header=True)
sibling_matrix.to_csv('sibling_matrix.csv', index=True, header=True)

def validate_hierarchy():
    all_nodes = set(child_to_parent.keys()) | set(child_to_parent.values())
    roots = all_nodes - set(child_to_parent.keys())  #nodes with no parent

    print(f"Number of roots: {len(roots)} — {roots}")
    assert len(roots) == 1, "Not a tree: expected exactly one root"

    # Cycle check via DFS
    def has_cycle(node, visited=set(), stack=set()):
        visited.add(node)
        stack.add(node)
        parent = child_to_parent.get(node)
        if parent is not None:
            if parent not in visited:
                if has_cycle(parent, visited, stack):
                    return True
            elif parent in stack:
                return True
        stack.discard(node)
        return False

    visited, stack = set(), set()
    for node in all_nodes:
        if node not in visited:
            assert not has_cycle(node, visited, stack), f"Cycle detected near node {node}"
    print("No cycles detected")

    #2. Check min and max depth are both 4
    def depth(node):
        d = 0
        current = node
        while current in child_to_parent:
            current = child_to_parent[current]
            d += 1
        return d

    leaf_nodes = all_nodes - set(child_to_parent.values())  # nodes with no children
    depths = {node: depth(node) for node in leaf_nodes}

    min_depth = min(depths.values())
    max_depth = max(depths.values())
    print(f"Leaf depths — min: {min_depth}, max: {max_depth}")

    if min_depth != 4 or max_depth != 4:
        # Show offending leaves to help debug
        bad = {n: d for n, d in depths.items() if d != 4}
        print(f"Leaves with wrong depth: {bad}")

    # assert min_depth == 4 and max_depth == 4, "Not all leaves are at depth 4"
    # print("All leaves are at depth 4")

validate_hierarchy()

with open('classes_birds.txt') as f:
    bird_names = [line.strip() for line in f]

all_nodes = set(child_to_parent.keys()) | set(child_to_parent.values())
leaf_nodes = all_nodes - set(child_to_parent.values())

depth3, depth4 = [], []
for node in leaf_nodes:
    d = 0
    cur = node
    while cur in child_to_parent:
        cur = child_to_parent[cur]
        d += 1
    (depth3 if d == 3 else depth4).append(node)

print("Depth-3 leaves:")
for n in sorted(depth3)[:10]:
    print(f"  {n}: {bird_names[n]}")

print("\nDepth-4 leaves:")
for n in sorted(depth4)[:10]:
    print(f"  {n}: {bird_names[n]}")

# Print the full path from root to a few leaves
def get_path(node):
    path = [node]
    while node in child_to_parent:
        node = child_to_parent[node]
        path.append(node)
    return list(reversed(path))

for leaf in list(leaf_nodes)[:5]:
    path = get_path(leaf)
    print(" -> ".join(f"{n}:{bird_names[n]}" for n in path))

print("DEPTH-3 LEAF PATHS")
for leaf in sorted(depth3)[:20]:
    path = get_path(leaf)
    print(" -> ".join(f"{n}:{bird_names[n]}" for n in path))

print("\nDEPTH-4 LEAF PATHS")
for leaf in sorted(depth4)[:20]:
    path = get_path(leaf)
    print(" -> ".join(f"{n}:{bird_names[n]}" for n in path))

#print a summary of what the intermediate nodes look like at each depth
print("\nINTERNAL NODE NAMES BY DEPTH")
depth_to_names = defaultdict(set)
for node in all_nodes:
    path = get_path(node)
    d = len(path) - 1
    depth_to_names[d].add(f"{node}:{bird_names[node]}")

for d in sorted(depth_to_names):
    samples = list(depth_to_names[d])[:8]
    print(f"Depth {d}: {samples}")