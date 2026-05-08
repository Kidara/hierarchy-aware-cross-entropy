# ALL HACE LOSS FUNCTIONS USED IN TRAINING
# Definitions:
# R: reachability matrix (N x N), R[i,j] = 1 if i is ancestor of j or i == j — Section 3.1, Eq. 2
# T: ancestral label smoothing matrix (N x n), T[i] = soft target for leaf class i — Section 3.2, Eq. 4
# Both are precomputed per dataset — see train_models_*/data_preprocessing.py


# HACE — Section 3
def hace_loss(logits, targets, R, T):
    #aggregate predicted probabilities upward through the hierarchy (Section 3.1, Eq. 2)
    agg_probs = torch.softmax(logits, dim=-1)
    agg_probs = torch.matmul(agg_probs, R.T)
    log_agg_probs = torch.log(agg_probs + 1e-6)

    #index into T to retrieve the ancestral label smoothing vector p* for each target (Section 3.2, Eq. 4)
    y_node = T[:, targets].T
    loss = -(y_node * log_agg_probs).sum(dim=1).mean()
    return loss


# HACE Smooth — Section 3.2, Step 1 (label smoothing) + Step 2 (ancestral label smoothing)
def hace_smooth_loss(logits, targets, R, T, epsilon, num_leaf_classes):
    #aggregate predicted probabilities upward through the hierarchy (Section 3.1, Eq. 2)
    agg_probs = torch.softmax(logits, dim=-1)
    agg_probs = torch.matmul(agg_probs, R.T)
    log_agg_probs = torch.log(agg_probs + 1e-6)

    #apply standard label smoothing horizontally over leaf classes (Section 3.2, Step 1, Eq. 3)
    batch_size = targets.size(0)
    smoothed = torch.full((batch_size, num_leaf_classes), epsilon / (num_leaf_classes - 1), device=logits.device)
    smoothed.scatter_(1, targets.unsqueeze(1), 1 - epsilon)

    #propagate smoothed leaf distribution upward through T (Section 3.2, Step 2, Eq. 4)
    y_node = torch.matmul(smoothed, T.T)
    loss = -(y_node * log_agg_probs).sum(dim=1).mean()
    return loss


# HACE with soft labels — Section 3.2, Step 1 (LCA-based soft targets) + Step 2 (ancestral label smoothing)
def hace_soft_labels_loss(logits, targets, R, T, soft_targets, orig_to_leaf, leaf_tensor, N):
    #aggregate predicted probabilities upward through the hierarchy (Section 3.1, Eq. 2)
    agg_probs = torch.softmax(logits, dim=-1)
    agg_probs = torch.matmul(agg_probs, R.T)
    log_agg_probs = torch.log(agg_probs + 1e-6)

    #soft_targets is defined over n leaves only; convert N-space node IDs to leaf indices (0 to n-1) to index into it
    leaf_idx = orig_to_leaf[targets.cpu()].to(logits.device)
    soft_leaf_dist = soft_targets[leaf_idx]

    #scatter soft leaf distribution back into N-space before ancestral smoothing
    smoothed = torch.zeros(targets.size(0), N, device=logits.device)
    smoothed[:, leaf_tensor] = soft_leaf_dist

    #propagate LCA-based soft distribution upward through T (Section 3.2, Step 2, Eq. 4)
    y_node = torch.matmul(smoothed, T.T)
    return -(y_node * log_agg_probs).sum(dim=1).mean()