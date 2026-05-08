"""
build_hxe_masks.py

Generates the three files required by fine_tune_hxe.py:
  child_leaf_masks.pt   [N_LEAVES, E, N_LEAVES]  float32
  parent_leaf_masks.pt  [N_LEAVES, E, N_LEAVES]  float32
  depth_matrix.csv      [N_LEAVES, E]             int  (depth of child node on each path edge)
  leaf_index.csv        [N_LEAVES]                int  (original node ID for each leaf index)

Inputs (already produced by your preprocessing script):
  hierarchy.txt          child -> parent mapping
  hierarchy_matrix.csv   [all_nodes x all_nodes], entry [i,j]=1 if j is in subtree of i

N_TOTAL  = 1011  (total nodes in the hierarchy, including internal nodes)
N_LEAVES = number of leaf nodes the model predicts over (~555 for NABirds)
E        = max path length from any leaf to root (padded with zeros for shorter paths)

Key fix vs. previous version:
  - All mask dimensions use N_LEAVES, NOT N_TOTAL.
  - Leaves are re-indexed 0..N_LEAVES-1 via leaf_to_idx.
  - The softmax in training must be over N_LEAVES logits only.
"""

import math
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

HIERARCHY_FILE       = "hierarchy.txt"
HIERARCHY_MATRIX_CSV = "hierarchy_matrix.csv"
N_TOTAL              = 1011   # total nodes (leaves + internal); used only for reading inputs

child_to_parent = {}
with open(HIERARCHY_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            child, parent = int(parts[0]), int(parts[1])
            child_to_parent[child] = parent

all_nodes  = set(child_to_parent.keys()) | set(child_to_parent.values())
leaf_nodes = all_nodes - set(child_to_parent.values())   # nodes that are never a parent

# Assign contiguous indices 0..N_LEAVES-1 to leaf nodes
leaf_list   = sorted(leaf_nodes)          # deterministic ordering
N_LEAVES    = len(leaf_list)
leaf_to_idx = {node: idx for idx, node in enumerate(leaf_list)}

print(f"Total nodes: {len(all_nodes)},  Leaf nodes: {N_LEAVES}")

H = pd.read_csv(HIERARCHY_MATRIX_CSV, index_col=0)
H.index   = H.index.astype(int)
H.columns = H.columns.astype(int)
def subtree_leaf_mask(node):
    """
    Returns a length-N_LEAVES float32 array where index j is 1.0 iff
    leaf_list[j] is a descendant of `node` (including node itself if it is a leaf).
    """
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
    """Returns [node, parent, grandparent, ..., root] (leaf first)."""
    path = [node]
    while node in child_to_parent:
        node = child_to_parent[node]
        path.append(node)
    return path

def node_depth(node):
    """Depth from root (root = 0)."""
    d = 0
    while node in child_to_parent:
        node = child_to_parent[node]
        d += 1
    return d

leaf_depths = {n: node_depth(n) for n in leaf_list}
E = max(leaf_depths.values())
print(f"Max leaf depth (= max edges E): {E}")

child_leaf_masks  = np.zeros((N_LEAVES, E, N_LEAVES), dtype=np.float32)
parent_leaf_masks = np.zeros((N_LEAVES, E, N_LEAVES), dtype=np.float32)
depth_matrix      = np.zeros((N_LEAVES, E),            dtype=np.int32)

for idx, leaf_orig_id in enumerate(leaf_list):
    path    = get_path_to_root(leaf_orig_id)   # [leaf, p1, p2, ..., root]
    n_edges = len(path) - 1

    for e in range(n_edges):
        child_node  = path[e]
        parent_node = path[e + 1]

        child_leaf_masks[idx,  e, :] = subtree_leaf_mask(child_node)
        parent_leaf_masks[idx, e, :] = subtree_leaf_mask(parent_node)
        depth_matrix[idx, e]         = node_depth(child_node)

print(f"child_leaf_masks shape:  {child_leaf_masks.shape}")   # [N_LEAVES, E, N_LEAVES]
print(f"parent_leaf_masks shape: {parent_leaf_masks.shape}")  # [N_LEAVES, E, N_LEAVES]
print(f"depth_matrix shape:      {depth_matrix.shape}")       # [N_LEAVES, E]


uniform = np.full(N_LEAVES, 1.0 / N_LEAVES, dtype=np.float32)

child_mass_all  = (child_leaf_masks  * uniform).sum(axis=-1)   # [N_LEAVES, E]
parent_mass_all = (parent_leaf_masks * uniform).sum(axis=-1)   # [N_LEAVES, E]

real_edges = parent_mass_all > 0

violations_child_parent = (child_mass_all > parent_mass_all + 1e-6) & real_edges
violations_parent_one   = (parent_mass_all > 1.0 + 1e-6)           & real_edges

if violations_child_parent.any():
    bad = np.argwhere(violations_child_parent)
    raise RuntimeError(f"child_mass > parent_mass at (leaf_idx, edge): {bad.tolist()}")

if violations_parent_one.any():
    bad = np.argwhere(violations_parent_one)
    raise RuntimeError(f"parent_mass > 1 at (leaf_idx, edge): {bad.tolist()}")

print("Mass conservation check passed.")

torch.save(torch.from_numpy(child_leaf_masks),  "child_leaf_masks.pt")
torch.save(torch.from_numpy(parent_leaf_masks), "parent_leaf_masks.pt")
pd.DataFrame(depth_matrix).to_csv("depth_matrix.csv", index=True, header=True)

# Save leaf_list so the training script knows the original-ID -> idx mapping
pd.DataFrame({"original_node_id": leaf_list}).to_csv("leaf_index.csv", index=True)

print(f"Saved: child_leaf_masks.pt, parent_leaf_masks.pt, depth_matrix.csv, leaf_index.csv")
print(f"N_LEAVES = {N_LEAVES}  <-- use this as N in fine_tune_hxe.py")

#soft labels
parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

def lca_depth(i, j):
    """Depth from root of the lowest common ancestor of i and j."""
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

# Build N_LEAVES x N_LEAVES distance matrix indexed by leaf_list order
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

pd.DataFrame(D_arr, index=leaf_list, columns=leaf_list).to_csv("distance_matrix.csv")
print(f"distance_matrix.csv shape: {D_arr.shape}")
leaf_set = set(leaf_list)

def build_soft_targets(dist_arr, beta):
    raw = np.exp(-beta * dist_arr)
    return raw / raw.sum(axis=1, keepdims=True)

def get_siblings_idx(ii):
    """Returns leaf indices (in 0..N_LEAVES-1) of siblings of leaf_list[ii]."""
    node = leaf_list[ii]
    if node not in child_to_parent:
        return []
    parent = child_to_parent[node]
    return [leaf_to_idx[c] for c in parent_to_children[parent]
            if c != node and c in leaf_to_idx]

def get_distant_idx(ii):
    """Returns leaf indices where lca with leaf_list[ii] is at depth 0 (root)."""
    node = leaf_list[ii]
    return [jj for jj, j in enumerate(leaf_list) if j != node and lca_depth(node, j) == 0]

TEST_BETA = 10
soft = build_soft_targets(D_arr, TEST_BETA)

print("\n--- Soft Labels validation ---")

# Check 1: rows sum to 1
row_sums = soft.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), \
    f"Rows do not sum to 1. Min={row_sums.min():.6f}, Max={row_sums.max():.6f}"
print(f"[PASS] Check 1: All rows sum to 1 (beta={TEST_BETA})")

# Check 2: diagonal is maximum
for ii in range(N_LEAVES):
    row = soft[ii]
    assert row[ii] == row.max(), \
        f"Diagonal not max for leaf_idx={ii} (orig={leaf_list[ii]})"
print("[PASS] Check 2: Diagonal is maximum in every row")

# Check 3: sibling preference
failures = []
for ii in range(min(50, N_LEAVES)):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    sibling_mass = np.mean([soft[ii, s] for s in sib_idxs])
    distant_mass = np.mean([soft[ii, d] for d in dis_idxs])
    if sibling_mass <= distant_mass:
        failures.append((ii, sibling_mass, distant_mass))
assert not failures, f"Sibling preference violated: {failures}"
print("[PASS] Check 3: Siblings receive more mass than distant classes")

# Check 4a: beta=0 -> uniform
soft_b0 = build_soft_targets(D_arr, beta=0)
assert np.allclose(soft_b0, 1.0 / N_LEAVES, atol=1e-6), "beta=0 not uniform"
print("[PASS] Check 4a: beta=0 produces uniform target")

# Check 4b: large beta -> one-hot
soft_large = build_soft_targets(D_arr, beta=1000)
for ii in range(min(10, N_LEAVES)):
    assert soft_large[ii, ii] >= soft_large[ii].max() - 1e-9, \
        f"beta=1000: diagonal not max for leaf_idx={ii}"
print("[PASS] Check 4b: Large beta gives near-one-hot target")

# Check 5: loss ordering
def ce_loss_soft(target_row, predicted_idx):
    log_p = np.full(N_LEAVES, -1e9)
    log_p[predicted_idx] = 0.0
    log_p -= math.log(np.exp(log_p).sum())
    return -(target_row * log_p).sum()

failures = []
for ii in range(min(50, N_LEAVES)):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    t = soft[ii]
    lc = ce_loss_soft(t, ii)
    ls = min(ce_loss_soft(t, s) for s in sib_idxs)
    ld = min(ce_loss_soft(t, d) for d in dis_idxs)
    if not (lc < ls < ld):
        failures.append((ii, lc, ls, ld))
assert not failures, f"Loss ordering violated: {failures[:3]}"
print(f"[PASS] Check 5: loss(correct) < loss(sibling) < loss(distant)  (beta={TEST_BETA})")

print("\n--- Distance matrix structural checks ---")
diag = np.diag(D_arr)
assert np.allclose(diag, 0.0, atol=1e-9), "Diagonal not zero"
print("[PASS] Same-class distance is 0")

assert np.allclose(D_arr, D_arr.T, atol=1e-9), "Distance matrix not symmetric"
print("[PASS] Distance matrix is symmetric")

for ii in range(min(50, N_LEAVES)):
    sib_idxs = get_siblings_idx(ii)
    dis_idxs = get_distant_idx(ii)
    if not sib_idxs or not dis_idxs:
        continue
    max_sib = max(D_arr[ii, s] for s in sib_idxs)
    min_dis = min(D_arr[ii, d] for d in dis_idxs)
    assert max_sib < min_dis, \
        f"Monotonicity violated for leaf_idx={ii}: sib={max_sib:.4f}, distant={min_dis:.4f}"
print("[PASS] Monotonicity: siblings closer than distant classes")

print("\nAll checks passed.")
print(f"\nSummary: N_LEAVES={N_LEAVES}. Update fine_tune_hxe.py to use N={N_LEAVES}.")