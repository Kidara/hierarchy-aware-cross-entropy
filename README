# Hierarchy-Aware Cross-Entropy (HACE)

A drop-in replacement for standard cross-entropy that incorporates a known class hierarchy 
directly into the loss. See `HACE_loss.py` for the core implementation.

## Repository Structure
- `HACE_loss.py` — all HACE loss functions
- `train_models_cifar100/` — training scripts for CIFAR-100
- `train_models_FGVC/` — training scripts for FGVC Aircraft
- `train_models_NABirds/` — training scripts for NABirds

Each dataset folder contains:
- `data_preprocessing.py` — builds R and T for that dataset; dilution value d is set via the `DILUTION` global variable at the top of the file
- `all_end_to_end_models/` — end-to-end training scripts
- `fine_tune_models/` — linear probing on frozen DINOv2-Large features
- `accuracy_analysis/` — evaluation scripts
- `data_preprocessing_matrices/` — precomputed R and T matrices for d ∈ {0.2, 0.5, 0.7}

## Reproducibility
Training was run on a private cluster with hardcoded data paths. To reproduce, 
update the data paths in the relevant training script and run `data_preprocessing.py` 
for the desired dataset to generate R and T before training.
