[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2605.06274-b31b1b.svg)](https://arxiv.org/abs/2605.06274)

# Hierarchy-Aware Cross-Entropy (HACE)

Official implementation of **"When Labels Have Structure: Improving Image Classification with Hierarchy-Aware Cross-Entropy"**.

> Standard cross-entropy is the default classification loss across virtually all of machine learning, yet it treats all misclassifications equally, ignoring the semantic distances that a class hierarchy encodes. We propose Hierarchy-Aware Cross-Entropy (HACE), a drop-in replacement for standard cross-entropy that incorporates a known class hierarchy directly into the loss. HACE combines two components: prediction aggregation, which propagates the model's probability mass upward through the class hierarchy to ensure that parent nodes accumulate the confidence of their children; and ancestral label smoothing, which distributes the ground-truth signal along the path from the true class to the root. We evaluate HACE on CIFAR-100, FGVC Aircraft, and NABirds in two regimes: end-to-end training across six architectures spanning convolutional and attention-based designs, and linear probing on frozen DINOv2-Large features. In end-to-end training, HACE improves accuracy over standard cross-entropy in 15 out of 18 architecture–dataset pairs, with a mean gain of 4.66%. In linear probing on frozen DINOv2-Large features, HACE outperforms all competing methods on all three datasets, with a mean improvement of 2.18% over the next best baseline.

---

## 📄 Citation

```bibtex
@article{chan2026hace,
  title   = {When Labels Have Structure: Improving Image Classification with Hierarchy-Aware Cross-Entropy},
  author  = {April Chan and Davide D'Ascenzo and Sebastiano Cultrera di Montesano},
  journal = {arXiv preprint arXiv:2605.06274},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.06274}
}
```

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
