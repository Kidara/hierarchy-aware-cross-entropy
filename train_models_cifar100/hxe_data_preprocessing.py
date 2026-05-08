"""
cifar100_build_hxe_masks.py

Generates files required by cifar100_fine_tune_hxe.py and cifar100_fine_tune_softlabels.py:
  cifar100_child_leaf_masks.pt      [100, E, 100]  float32
  cifar100_parent_leaf_masks.pt     [100, E, 100]  float32
  cifar100_depth_matrix.csv         [100, E]        int
  cifar100_leaf_index.csv           [100]            int  (hierarchy node ID per leaf_idx)
  cifar100_distance_matrix.csv      [100, 100]       float64
  cifar100_sibling_matrix.csv       [100, 100]       int

CIFAR-100 hierarchy:
  Fine classes  : dataset labels 0-99  -> hierarchy node IDs 20-119
  Coarse classes: hierarchy node IDs 0-19
  Virtual root  : node ID 120  (added so the hierarchy is a proper single-rooted tree)

Leaf nodes are the 100 fine classes. All leaves are at depth 2:
  fine (depth 2) -> coarse (depth 1) -> root (depth 0)
So E = 2, and there are exactly 2 path edges per leaf.

Edge 0 (fine -> coarse):
  child_mass  = p(true fine class)
  parent_mass = p(all fine classes in same coarse group)

Edge 1 (coarse -> root):
  child_mass  = p(all fine classes in same coarse group)
  parent_mass = p(all 100 fine classes) = 1

Distance:
  d(i, j) = 1 - lca_depth(i, j) / max_leaf_depth
  same class   -> 0
  same coarse  -> 1 - 1/2 = 0.5
  diff coarse  -> 1 - 0/2 = 1.0
"""

import math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Reproduce hierarchy from fine_to_coarse mapping (same as data_preprocessing.py)
# ---------------------------------------------------------------------------
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

meta             = unpickle('cifar-100-python/meta')
fine_label_names = [l.decode('utf-8') for l in meta[b'fine_label_names']]

print(f"Loaded {len(fine_label_names)} fine label names from meta file.")
assert len(fine_label_names) == 100, f"Expected 100 fine labels, got {len(fine_label_names)}"

fine_to_coarse = {
    "beaver": "aquatic_mammals", "dolphin": "aquatic_mammals", "otter": "aquatic_mammals",
    "seal": "aquatic_mammals", "whale": "aquatic_mammals",
    "aquarium_fish": "fish", "flatfish": "fish", "ray": "fish", "shark": "fish", "trout": "fish",
    "orchid": "flowers", "poppy": "flowers", "rose": "flowers", "sunflower": "flowers", "tulip": "flowers",
    "bottle": "food_containers", "bowl": "food_containers", "can": "food_containers",
    "cup": "food_containers", "plate": "food_containers",
    "apple": "fruit_and_vegetables", "mushroom": "fruit_and_vegetables", "orange": "fruit_and_vegetables",
    "pear": "fruit_and_vegetables", "sweet_pepper": "fruit_and_vegetables",
    "clock": "household_electrical_devices", "keyboard": "household_electrical_devices",
    "lamp": "household_electrical_devices", "telephone": "household_electrical_devices",
    "television": "household_electrical_devices",
    "bed": "household_furniture", "chair": "household_furniture", "couch": "household_furniture",
    "table": "household_furniture", "wardrobe": "household_furniture",
    "bee": "insects", "beetle": "insects", "butterfly": "insects", "caterpillar": "insects",
    "cockroach": "insects",
    "bear": "large_carnivores", "leopard": "large_carnivores", "lion": "large_carnivores",
    "tiger": "large_carnivores", "wolf": "large_carnivores",
    "bridge": "large_man-made_outdoor_things", "castle": "large_man-made_outdoor_things",
    "house": "large_man-made_outdoor_things", "road": "large_man-made_outdoor_things",
    "skyscraper": "large_man-made_outdoor_things",
    "cloud": "large_natural_outdoor_scenes", "forest": "large_natural_outdoor_scenes",
    "mountain": "large_natural_outdoor_scenes", "plain": "large_natural_outdoor_scenes",
    "sea": "large_natural_outdoor_scenes",
    "camel": "large_omnivores_and_herbivores", "cattle": "large_omnivores_and_herbivores",
    "chimpanzee": "large_omnivores_and_herbivores", "elephant": "large_omnivores_and_herbivores",
    "kangaroo": "large_omnivores_and_herbivores",
    "fox": "medium_mammals", "porcupine": "medium_mammals", "possum": "medium_mammals",
    "raccoon": "medium_mammals", "skunk": "medium_mammals",
    "crab": "non-insect_invertebrates", "lobster": "non-insect_invertebrates",
    "snail": "non-insect_invertebrates", "spider": "non-insect_invertebrates",
    "worm": "non-insect_invertebrates",
    "baby": "people", "boy": "people", "girl": "people", "man": "people", "woman": "people",
    "crocodile": "reptiles", "dinosaur": "reptiles", "lizard": "reptiles",
    "snake": "reptiles", "turtle": "reptiles",
    "hamster": "small_mammals", "mouse": "small_mammals", "rabbit": "small_mammals",
    "shrew": "small_mammals", "squirrel": "small_mammals",
    "maple_tree": "trees", "oak_tree": "trees", "palm_tree": "trees",
    "pine_tree": "trees", "willow_tree": "trees",
    "bicycle": "vehicles_1", "bus": "vehicles_1", "motorcycle": "vehicles_1",
    "pickup_truck": "vehicles_1", "train": "vehicles_1",
    "lawn_mower": "vehicles_2", "rocket": "vehicles_2", "streetcar": "vehicles_2",
    "tank": "vehicles_2", "tractor": "vehicles_2",
}

coarse_label_names = sorted(set(fine_to_coarse.values()))
coarse_name_to_id  = {name: i for i, name in enumerate(coarse_label_names)}

FINE_OFFSET   = 20
COARSE_OFFSET = 0
ROOT_ID       = 120
N_LEAVES      = 100
N_TOTAL       = 121 

child_to_parent = {}
for fine_idx, fine_name in enumerate(fine_label_names):
    fine_node   = fine_idx + FINE_OFFSET
    coarse_node = coarse_name_to_id[fine_to_coarse[fine_name]]
    child_to_parent[fine_node] = coarse_node

for coarse_id in range(20):
    child_to_parent[coarse_id] = ROOT_ID


leaf_list   = list(range(FINE_OFFSET, FINE_OFFSET + N_LEAVES))
leaf_to_idx = {node: idx for idx, node in enumerate(leaf_list)}

print(f"N_LEAVES={N_LEAVES}, N_TOTAL={N_TOTAL}, ROOT_ID={ROOT_ID}")
print(f"leaf_list[0]={leaf_list[0]}, leaf_list[-1]={leaf_list[-1]}")


def node_depth(node):
    """Depth from root (root=0, coarse=1, fine=2)."""
    d = 0
    while node in child_to_parent:
        node = child_to_parent[node]
        d += 1
    return d

def get_path_to_root(node):
    path = [node]
    while node in child_to_parent:
        node = child_to_parent[node]
        path.append(node)
    return path

def subtree_leaves(node):
    """All leaf_idxs (0..99) that are descendants of node."""
    if node in leaf_to_idx:
        return [leaf_to_idx[node]]
    result = []
    for child, parent in child_to_parent.items():
        if parent == node and child in leaf_to_idx:
            result.append(leaf_to_idx[child])
        elif parent == node:
            result.extend(subtree_leaves(child))
    return result

parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

def subtree_leaf_mask(node):
    """Length-100 float32 mask; 1.0 for every leaf under node."""
    mask = np.zeros(N_LEAVES, dtype=np.float32)
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in leaf_to_idx:
            mask[leaf_to_idx[cur]] = 1.0
        else:
            stack.extend(parent_to_children[cur])
    return mask

leaf_depths = {n: node_depth(n) for n in leaf_list}
E = max(leaf_depths.values())
assert E == 2, (f"Expected E=2 (fine->coarse->root), got E={E}. Check all fine_label_names are in fine_to_coarse.")
bad_depths = {n: d for n, d in leaf_depths.items() if d != 2}
assert not bad_depths, f"Fine nodes not at depth 2: {bad_depths}. Check fine_to_coarse coverage."
print(f"All leaves at depth 2, E={E} (fine->coarse->root)")

child_leaf_masks  = np.zeros((N_LEAVES, E, N_LEAVES), dtype=np.float32)
parent_leaf_masks = np.zeros((N_LEAVES, E, N_LEAVES), dtype=np.float32)
depth_matrix      = np.zeros((N_LEAVES, E),            dtype=np.int32)

for idx, leaf_orig_id in enumerate(leaf_list):
    path    = get_path_to_root(leaf_orig_id)
    n_edges = len(path) - 1

    for e in range(n_edges):
        child_node  = path[e]
        parent_node = path[e + 1]
        child_leaf_masks[idx,  e, :] = subtree_leaf_mask(child_node)
        parent_leaf_masks[idx, e, :] = subtree_leaf_mask(parent_node)
        depth_matrix[idx, e]         = node_depth(child_node)

print(f"child_leaf_masks:  {child_leaf_masks.shape}")
print(f"parent_leaf_masks: {parent_leaf_masks.shape}")
print(f"depth_matrix:      {depth_matrix.shape}")

assert child_leaf_masks[0, 0, :].sum() == 1.0, "Edge-0 child mask should be one-hot"
assert parent_leaf_masks[0, 0, :].sum() == 5.0, "Edge-0 parent mask should have 5 leaves"
assert parent_leaf_masks[0, 1, :].sum() == 100.0, "Edge-1 parent mask should cover all 100 leaves"
print("Spot-checks passed.")

uniform = np.full(N_LEAVES, 1.0 / N_LEAVES, dtype=np.float32)
child_mass_all  = (child_leaf_masks  * uniform).sum(axis=-1)
parent_mass_all = (parent_leaf_masks * uniform).sum(axis=-1)
real_edges      = parent_mass_all > 0

if ((child_mass_all > parent_mass_all + 1e-6) & real_edges).any():
    bad = np.argwhere((child_mass_all > parent_mass_all + 1e-6) & real_edges)
    raise RuntimeError(f"child_mass > parent_mass at {bad.tolist()}")
if ((parent_mass_all > 1.0 + 1e-6) & real_edges).any():
    bad = np.argwhere((parent_mass_all > 1.0 + 1e-6) & real_edges)
    raise RuntimeError(f"parent_mass > 1 at {bad.tolist()}")
print("Mass conservation check passed.")

torch.save(torch.from_numpy(child_leaf_masks),  "cifar100_child_leaf_masks.pt")
torch.save(torch.from_numpy(parent_leaf_masks), "cifar100_parent_leaf_masks.pt")
pd.DataFrame(depth_matrix).to_csv("cifar100_depth_matrix.csv", index=True, header=True)
pd.DataFrame({"original_node_id": leaf_list}).to_csv("cifar100_leaf_index.csv", index=True)
print("Saved HXE mask files.")

sibling_matrix = np.zeros((N_LEAVES, N_LEAVES), dtype=np.int32)
for coarse_id in range(20):
    fine_children = [leaf_to_idx[c] for c in parent_to_children[coarse_id] if c in leaf_to_idx]
    for i in fine_children:
        for j in fine_children:
            sibling_matrix[i, j] = 1

pd.DataFrame(sibling_matrix).to_csv("cifar100_sibling_matrix.csv", index=True, header=True)
print(f"cifar100_sibling_matrix.csv: {sibling_matrix.shape}")

max_leaf_depth = 2

def lca_depth(i, j):
    """Depth from root of LCA of leaf_list[i] and leaf_list[j]."""
    ni, nj = leaf_list[i], leaf_list[j]
    ancestors_i = set()
    cur = ni
    while True:
        ancestors_i.add(cur)
        if cur not in child_to_parent:
            break
        cur = child_to_parent[cur]
    cur = nj
    while True:
        if cur in ancestors_i:
            return node_depth(cur)
        if cur not in child_to_parent:
            break
        cur = child_to_parent[cur]
    return 0

D_arr = np.zeros((N_LEAVES, N_LEAVES), dtype=np.float64)
for ii in range(N_LEAVES):
    for jj in range(ii, N_LEAVES):
        if ii == jj:
            D_arr[ii, jj] = 0.0
        else:
            d = 1.0 - lca_depth(ii, jj) / max_leaf_depth
            D_arr[ii, jj] = d
            D_arr[jj, ii] = d

pd.DataFrame(D_arr).to_csv("cifar100_distance_matrix.csv", index=True, header=True)
print(f"cifar100_distance_matrix.csv: {D_arr.shape}")

same_coarse_pairs = [(i, j) for i in range(N_LEAVES) for j in range(i+1, N_LEAVES)
                     if sibling_matrix[i, j] == 1]
diff_coarse_pairs = [(i, j) for i in range(N_LEAVES) for j in range(i+1, N_LEAVES)
                     if sibling_matrix[i, j] == 0]
assert all(abs(D_arr[i, j] - 0.5) < 1e-9 for i, j in same_coarse_pairs[:20]), \
    "Same-coarse distance should be 0.5"
assert all(abs(D_arr[i, j] - 1.0) < 1e-9 for i, j in diff_coarse_pairs[:20]), \
    "Diff-coarse distance should be 1.0"
print("[PASS] Distance values: same-coarse=0.5, diff-coarse=1.0")

def build_soft_targets(dist_arr, beta):
    raw = np.exp(-beta * dist_arr)
    return raw / raw.sum(axis=1, keepdims=True)

TEST_BETA = 10
soft = build_soft_targets(D_arr, TEST_BETA)

print("\n--- Soft Labels validation ---")

assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6), "Rows not summing to 1"
print(f"[PASS] Check 1: All rows sum to 1 (beta={TEST_BETA})")

for ii in range(N_LEAVES):
    assert soft[ii, ii] == soft[ii].max(), f"Diagonal not max for leaf_idx={ii}"
print("[PASS] Check 2: Diagonal is maximum in every row")

for ii in range(N_LEAVES):
    sib_mass = np.mean([soft[ii, jj] for jj in range(N_LEAVES)
                        if jj != ii and sibling_matrix[ii, jj] == 1])
    dis_mass = np.mean([soft[ii, jj] for jj in range(N_LEAVES)
                        if sibling_matrix[ii, jj] == 0])
    assert sib_mass > dis_mass, f"Sibling preference violated for leaf_idx={ii}"
print("[PASS] Check 3: Siblings receive more mass than distant classes")

soft_b0 = build_soft_targets(D_arr, beta=0)
assert np.allclose(soft_b0, 1.0 / N_LEAVES, atol=1e-6), "beta=0 not uniform"
print("[PASS] Check 4a: beta=0 produces uniform target")

soft_large = build_soft_targets(D_arr, beta=1000)
for ii in range(N_LEAVES):
    assert soft_large[ii, ii] >= soft_large[ii].max() - 1e-9
print("[PASS] Check 4b: Large beta gives near-one-hot target")

print("\nAll checks passed.")
print(f"\nSummary: N_LEAVES={N_LEAVES}, E={E}")
print("Dataset fine_labels (0-99) = leaf indices directly (no remapping needed).")