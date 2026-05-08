[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2605.06274-b31b1b.svg)](https://arxiv.org/abs/2605.06274)

# HACE: Hierarchy-Aware Cross-Entropy

Official PyTorch implementation of **"When Labels Have Structure: Improving Image Classification with Hierarchy-Aware Cross-Entropy"**.

<p align="center">
  <img src="PandQ.png" width="600" alt="P and Q"/>
</p>

---

## 🔑 Key Ideas

- **Prediction Aggregation** — probabilities are propagated up the class hierarchy, penalizing mistakes more heavily when predicted and true classes are semantically distant.
- **Ancestral Label Smoothing** — the ground-truth label mass is redistributed upward along the path from the true leaf class to the root, with each ancestor receiving geometrically decaying weight controlled by a dilution parameter *d*.
- **Drop-in replacement** — HACE replaces `nn.CrossEntropyLoss` with no changes to model architecture or inference pipeline.

---

## 📁 Repository Structure

- `HACE_loss.py` — all HACE loss functions
- `train_models_cifar100/` — training scripts for CIFAR-100
- `train_models_FGVC/` — training scripts for FGVC Aircraft
- `train_models_NABirds/` — training scripts for NABirds

Each dataset folder contains:

- `data_preprocessing.py` — builds **R** and **T** for that dataset; dilution value *d* is set via the `DILUTION` global variable at the top of the file
- `all_end_to_end_models/` — end-to-end training scripts
- `fine_tune_models/` — linear probing on frozen DINOv2-Large features
- `accuracy_analysis/` — evaluation scripts
- `data_preprocessing_matrices/` — precomputed **R** and **T** matrices for *d* ∈ {0.2, 0.5, 0.7}

---

## 🔁 Reproducibility

Training was run on a private cluster with hardcoded data paths. To reproduce, update the data paths in the relevant training script and run `data_preprocessing.py` for the desired dataset to generate **R** and **T** before training.

---

## 📄 Paper

If you use this work, please cite:

```bibtex
@article{chan2026hace,
  title         = {When Labels Have Structure: Improving Image Classification with Hierarchy-Aware Cross-Entropy},
  author        = {April Chan and Davide D'Ascenzo and Sebastiano Cultrera di Montesano},
  journal       = {arXiv preprint arXiv:2605.06274},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.06274}
}
```

---


