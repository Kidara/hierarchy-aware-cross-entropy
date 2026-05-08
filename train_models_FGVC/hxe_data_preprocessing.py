"""
fgvc_build_hxe_masks.py

Generates files required by fgvc_fine_tune_hxe.py and fgvc_fine_tune_softlabels.py:
  child_leaf_masks.pt    [N_LEAVES, E, N_LEAVES]  float32
  parent_leaf_masks.pt   [N_LEAVES, E, N_LEAVES]  float32
  depth_matrix.csv       [N_LEAVES, E]             int
  fgvc_leaf_index.csv    [N_LEAVES]                int  (original node ID per leaf idx)
  fgvc_distance_matrix.csv  [N_LEAVES, N_LEAVES]   float64

FGVC hierarchy is 3 levels deep: variant -> family -> manufacturer -> root
  N_LEAVES = 102  (variants, original IDs 0..101)
  N_TOTAL  = 201  (variants + families + manufacturers + root)
  E        = 3    (all leaves are at depth 3, so path has exactly 3 edges)
"""

import math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

HIERARCHY_FILE       = "fgvc_hierarchy.txt"
HIERARCHY_MATRIX_CSV = "fgvc_hierarchy_matrix.csv"
N_TOTAL              = 201

child_to_parent = {}
with open(HIERARCHY_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            child, parent = int(parts[0]), int(parts[1])
            child_to_parent[child] = parent

all_nodes  = set(child_to_parent.keys()) | set(child_to_parent.values())
leaf_nodes = all_nodes - set(child_to_parent.values())

leaf_list   = sorted(leaf_nodes)          # deterministic; should be [0, 1, ..., 101]
N_LEAVES    = len(leaf_list)
leaf_to_idx = {node: idx for idx, node in enumerate(leaf_list)}

print(f"Total nodes: {len(all_nodes)},  Leaf nodes: {N_LEAVES}")
print(f"N_LEAVES = {N_LEAVES}")
H = pd.read_csv(HIERARCHY_MATRIX_CSV, index_col=0)
H.index   = H.index.astype(int)
H.columns = H.columns.astype(int)

def subtree_leaf_mask(node):
    """Length-N_LEAVES float32 array; 1.0 where leaf is a descendant of node."""
    mask = np.zeros(N_LEAVES, dtype=np.float32)
    if node not in H.index:
        return mask
    row = H.loc[node]
    for orig_id, val in row.items():
        orig_id = int(orig_id)
        if val == 1 and orig_id in leaf_to_idx:
            mask[leaf_to_idx[orig_id]] = 1.0
    return mask

def get_path_to_root(node):
    path = [node]
    while node in child_to_parent:
        node = child_to_parent[node]
        path.append(node)
    return path  # [leaf, ..., root]

def node_depth(node):
    d = 0
    while node in child_to_parent:
        node = child_to_parent[node]
        d += 1
    return d

leaf_depths = {n: node_depth(n) for n in leaf_list}
E = max(leaf_depths.values())
print(f"Max leaf depth (= max edges E): {E}")
assert E == 3, f"Expected depth 3 for all FGVC leaves, got E={E}"

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

torch.save(torch.from_numpy(child_leaf_masks),  "fgvc_child_leaf_masks.pt")
torch.save(torch.from_numpy(parent_leaf_masks), "fgvc_parent_leaf_masks.pt")
pd.DataFrame(depth_matrix).to_csv("fgvc_depth_matrix.csv", index=True, header=True)
pd.DataFrame({"original_node_id": leaf_list}).to_csv("fgvc_leaf_index.csv", index=True)

print(f"Saved: fgvc_child_leaf_masks.pt, fgvc_parent_leaf_masks.pt, "
      f"fgvc_depth_matrix.csv, fgvc_leaf_index.csv")
print(f"N_LEAVES = {N_LEAVES}  <-- use this as N_LEAVES in training scripts")


parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

def lca_depth(i, j):
    ancestors_i = set()
    cur = i
    while True:
        ancestors_i.add(cur)
        if cur not in child_to_parent:
            break
        cur = child_to_parent[cur]
    cur = j
    while True:
        if cur in ancestors_i:
            return node_depth(cur)
        if cur not in child_to_parent:
            break
        cur = child_to_parent[cur]
    return 0

max_leaf_depth = max(leaf_depths.values())

D_arr = np.zeros((N_LEAVES, N_LEAVES), dtype=np.float64)
for ii, i in enumerate(leaf_list):
    for jj, j in enumerate(leaf_list):
        if i == j:
            D_arr[ii, jj] = 0.0
        elif jj < ii:
            D_arr[ii, jj] = D_arr[jj, ii]
        else:
            D_arr[ii, jj] = 1.0 - lca_depth(i, j) / max_leaf_depth
            D_arr[jj, ii] = D_arr[ii, jj]

pd.DataFrame(D_arr, index=leaf_list, columns=leaf_list).to_csv("fgvc_distance_matrix.csv")
print(f"fgvc_distance_matrix.csv shape: {D_arr.shape}")

leaf_set = set(leaf_list)

def build_soft_targets(dist_arr, beta):
    raw = np.exp(-beta * dist_arr)
    return raw / raw.sum(axis=1, keepdims=True)

def get_siblings_idx(ii):
    node = leaf_list[ii]
    if node not in child_to_parent:
        return []
    parent = child_to_parent[node]
    return [leaf_to_idx[c] for c in parent_to_children[parent]
            if c != node and c in leaf_to_idx]

def get_distant_idx(ii):
    node = leaf_list[ii]
    return [jj for jj, j in enumerate(leaf_list) if j != node and lca_depth(node, j) == 0]

TEST_BETA = 10
soft = build_soft_targets(D_arr, TEST_BETA)

print("\n--- Soft Labels validation ---")

row_sums = soft.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), f"Rows not summing to 1"
print(f"[PASS] Check 1: All rows sum to 1 (beta={TEST_BETA})")

for ii in range(N_LEAVES):
    assert soft[ii, ii] == soft[ii].max(), \
        f"Diagonal not max for leaf_idx={ii}"
print("[PASS] Check 2: Diagonal is maximum in every row")

failures = []
for ii in range(N_LEAVES):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    if np.mean([soft[ii, s] for s in sib_idxs]) <= np.mean([soft[ii, d] for d in dis_idxs]):
        failures.append(ii)
assert not failures, f"Sibling preference violated for {failures}"
print("[PASS] Check 3: Siblings receive more mass than distant classes")

soft_b0 = build_soft_targets(D_arr, beta=0)
assert np.allclose(soft_b0, 1.0 / N_LEAVES, atol=1e-6), "beta=0 not uniform"
print("[PASS] Check 4a: beta=0 produces uniform target")

soft_large = build_soft_targets(D_arr, beta=1000)
for ii in range(N_LEAVES):
    assert soft_large[ii, ii] >= soft_large[ii].max() - 1e-9
print("[PASS] Check 4b: Large beta gives near-one-hot target")

def ce_loss_soft(target_row, predicted_idx):
    log_p = np.full(N_LEAVES, -1e9)
    log_p[predicted_idx] = 0.0
    log_p -= math.log(np.exp(log_p).sum())
    return -(target_row * log_p).sum()

failures = []
for ii in range(N_LEAVES):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    t  = soft[ii]
    lc = ce_loss_soft(t, ii)
    ls = min(ce_loss_soft(t, s) for s in sib_idxs)
    ld = min(ce_loss_soft(t, d) for d in dis_idxs)
    if not (lc < ls < ld):
        failures.append((ii, lc, ls, ld))
assert not failures, f"Loss ordering violated: {failures[:3]}"
print(f"[PASS] Check 5: loss(correct) < loss(sibling) < loss(distant) (beta={TEST_BETA})")

print("\n--- Distance matrix structural checks ---")
assert np.allclose(np.diag(D_arr), 0.0, atol=1e-9), "Diagonal not zero"
print("[PASS] Same-class distance is 0")
assert np.allclose(D_arr, D_arr.T, atol=1e-9), "Not symmetric"
print("[PASS] Distance matrix is symmetric")

for ii in range(N_LEAVES):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    assert max(D_arr[ii, s] for s in sib_idxs) < min(D_arr[ii, d] for d in dis_idxs), \
        f"Monotonicity violated for leaf_idx={ii}"
print("[PASS] Monotonicity: siblings closer than distant classes")

print("\nAll checks passed.")