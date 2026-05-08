import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = '/data/user/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data'
DILUTION = 0.7

def load_label_list(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

variants      = load_label_list(f'{DATA_DIR}/variants.txt')
families      = load_label_list(f'{DATA_DIR}/families.txt')
manufacturers = load_label_list(f'{DATA_DIR}/manufacturers.txt')

VARIANT_OFFSET      = 0
FAMILY_OFFSET       = len(variants)
MANUFACTURER_OFFSET = FAMILY_OFFSET + len(families)
ROOT_IDX            = MANUFACTURER_OFFSET + len(manufacturers)
N                   = ROOT_IDX + 1

variant_to_idx      = {v: i + VARIANT_OFFSET      for i, v in enumerate(variants)}
family_to_idx       = {f: i + FAMILY_OFFSET        for i, f in enumerate(families)}
manufacturer_to_idx = {m: i + MANUFACTURER_OFFSET  for i, m in enumerate(manufacturers)}

def load_image_labels(path):
    mapping = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

variant_labels      = load_image_labels(f'{DATA_DIR}/images_variant_trainval.txt')
family_labels       = load_image_labels(f'{DATA_DIR}/images_family_trainval.txt')
manufacturer_labels = load_image_labels(f'{DATA_DIR}/images_manufacturer_trainval.txt')

variant_to_family = {}
for img_id, var in variant_labels.items():
    fam = family_labels.get(img_id)
    if var and fam:
        variant_to_family[var] = fam

family_to_manufacturer = {}
for img_id, fam in family_labels.items():
    mfr = manufacturer_labels.get(img_id)
    if fam and mfr:
        family_to_manufacturer[fam] = mfr

missing_v2f = [v for v in variants if v not in variant_to_family]
missing_f2m = [f for f in families if f not in family_to_manufacturer]
if missing_v2f:
    print(f"WARNING: {len(missing_v2f)} variants missing family mapping: {missing_v2f[:5]}")
if missing_f2m:
    print(f"WARNING: {len(missing_f2m)} families missing manufacturer mapping: {missing_f2m[:5]}")

child_to_parent = {}
for variant, family in variant_to_family.items():
    if variant in variant_to_idx and family in family_to_idx:
        child_to_parent[variant_to_idx[variant]] = family_to_idx[family]

for family, manufacturer in family_to_manufacturer.items():
    if family in family_to_idx and manufacturer in manufacturer_to_idx:
        child_to_parent[family_to_idx[family]] = manufacturer_to_idx[manufacturer]

for manufacturer_idx in range(MANUFACTURER_OFFSET, ROOT_IDX):
    child_to_parent[manufacturer_idx] = ROOT_IDX

with open('fgvc_hierarchy.txt', 'w') as f:
    for child, parent in sorted(child_to_parent.items()):
        f.write(f"{child} {parent}\n")

with open('fgvc_classes.txt', 'w') as f:
    for name, idx in variant_to_idx.items():
        f.write(f"{idx} {name}\n")
    for name, idx in family_to_idx.items():
        f.write(f"{idx} {name}\n")
    for name, idx in manufacturer_to_idx.items():
        f.write(f"{idx} {name}\n")
    f.write(f"{ROOT_IDX} ROOT\n")

hierarchy_matrix = pd.DataFrame(0,   index=range(N), columns=range(N), dtype=int)
sibling_matrix   = pd.DataFrame(0,   index=range(N), columns=range(N), dtype=int)
diluted_encoding = pd.DataFrame(0.0, index=range(N), columns=range(N), dtype=float)

def dfs_ancestors(current_node, leaf_node):
    if current_node not in child_to_parent:
        return
    parent = child_to_parent[current_node]
    hierarchy_matrix.loc[parent, current_node] = 1
    hierarchy_matrix.loc[parent, leaf_node]    = 1
    dfs_ancestors(parent, leaf_node)

def soft_one_hot(leaf, node, probability):
    if node not in child_to_parent:
        diluted_encoding.loc[node, leaf] += probability
        return
    diluted_encoding.loc[node, leaf] += DILUTION * probability
    parent = child_to_parent[node]
    soft_one_hot(leaf, parent, (1 - DILUTION) * probability)

for node in range(N):
    hierarchy_matrix.loc[node, node] = 1
    dfs_ancestors(node, node)
    soft_one_hot(node, node, 1)

parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

for parent, children in parent_to_children.items():
    for i in children:
        for j in children:
            sibling_matrix.loc[i, j] = 1

print(f"Sum of columns for soft one-hot encodings:\n{diluted_encoding.sum(axis=0)}")
hierarchy_matrix.to_csv('fgvc_hierarchy_matrix.csv', index=True, header=True)
diluted_encoding.to_csv('fgvc_diluted_encoding.csv', index=True, header=True)
sibling_matrix.to_csv(  'fgvc_sibling_matrix.csv',   index=True, header=True)

with open('fgvc_classes.txt') as f:
    fgvc_names = {}
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            fgvc_names[int(parts[0])] = parts[1]

all_nodes  = set(child_to_parent.keys()) | set(child_to_parent.values())
leaf_nodes = all_nodes - set(child_to_parent.values())

# def get_depth(node):
#     d = 0
#     cur = node
#     while cur in child_to_parent:
#         cur = child_to_parent[cur]
#         d += 1
#     return d

# def get_path(node):
#     path = [node]
#     while node in child_to_parent:
#         node = child_to_parent[node]
#         path.append(node)
#     return list(reversed(path))

# depth_counts = defaultdict(list)
# for node in leaf_nodes:
#     depth_counts[get_depth(node)].append(node)

# print("LEAF DEPTH DISTRIBUTION")
# for d in sorted(depth_counts):
#     print(f"Depth {d}: {len(depth_counts[d])} leaves")

# print("\nSAMPLE PATHS PER DEPTH")
# for d in sorted(depth_counts):
#     print(f"\n-- Depth {d} --")
#     for leaf in sorted(depth_counts[d])[:5]:
#         path = get_path(leaf)
#         print(" -> ".join(f"{n}:{fgvc_names.get(n, '?')}" for n in path))

# def has_cycle():
#     for start in all_nodes:
#         visited = set()
#         cur = start
#         while cur in child_to_parent:
#             if cur in visited:
#                 print(f"Cycle detected starting from node {start}, repeated node {cur}: {fgvc_names.get(cur, '?')}")
#                 return True
#             visited.add(cur)
#             cur = child_to_parent[cur]
#     return False

# if has_cycle():
#     print("CYCLES FOUND")
# else:
#     print("No cycles detected — hierarchy is a valid tree")