# Assessing the Beneficiality of Attention Rollout for Pooling in Vision Transformers

A comparative study of two pooling strategies for Vision Transformers — standard **[CLS] token pooling** vs. **attention-rollout pooling** — evaluated across CIFAR-100, CIFAR-10, and SVHN.

---

## Overview

Vision Transformers (ViTs) typically pool information from patch tokens using either the dedicated `[CLS]` token or, alternatively, attention-rollout weighted aggregation of patch tokens. This project implements and evaluates both strategies **under an identical backbone**, isolating the pooling mechanism as the sole variable, in order to answer:

> *Does attention-rollout pooling produce more discriminative, better-separated representations than [CLS] pooling — and does the answer depend on the dataset?*

## Key Features

- **Identical backbone** for both pooling strategies — ensures a fair, apples-to-apples comparison
- **Reproducible training pipeline**: AdamW optimizer, linear warm-up + cosine LR decay, automatic mixed precision (AMP)
- **Multi-dataset evaluation**: CIFAR-100, CIFAR-10, SVHN
- **Representation-quality metrics**:
  - k-NN accuracy
  - Intra-class distance
  - Inter-class distance
  - Silhouette score

## Pooling Strategies

| Strategy | Description |
|---|---|
| **[CLS] Pooling** | Standard approach — a learnable classification token attends over all patch tokens; its final-layer representation is used for downstream tasks. |
| **Attention Rollout Pooling** | Aggregates patch tokens by recursively multiplying attention maps across layers, producing a weighted pooling of patch representations that reflects cumulative attention flow. |

## Results Summary

| Dataset | Outcome |
|---|---|
| **CIFAR-10** | Rollout pooling improves class separability and convergence speed |
| **CIFAR-100** | Mixed results — better inter-class separation, but higher intra-class variance |
| **SVHN** | No clear benefit over [CLS] pooling |

### Takeaway

> Use **attention-rollout pooling** for cluttered scenes or images with variable backgrounds (e.g., CIFAR-style datasets).
> Stick to **[CLS] pooling** for simpler, centered-object datasets (e.g., SVHN).

## Evaluation Metrics

- **k-NN Accuracy** — classification accuracy using a k-nearest-neighbors classifier on learned embeddings
- **Intra-class Distance** — average distance between embeddings within the same class (lower = tighter clusters)
- **Inter-class Distance** — average distance between embeddings across classes (higher = better separation)
- **Silhouette Score** — combined measure of cohesion and separation of the embedding space

## Training Setup

- **Optimizer**: AdamW
- **LR Schedule**: Linear warm-up followed by cosine decay
- **Precision**: Automatic Mixed Precision (AMP)
- **Datasets**: CIFAR-100, CIFAR-10, SVHN

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Train with [CLS] pooling
python train.py --pooling cls --dataset cifar10

# Train with attention-rollout pooling
python train.py --pooling rollout --dataset cifar10
```

## Repository Structure

```
.
├── data/               # Dataset loading and preprocessing
├── models/             # ViT backbone and pooling implementations
├── train.py            # Training entry point
├── evaluate.py         # k-NN, distance, and silhouette evaluation
├── configs/             # Experiment configs per dataset
└── README.md
```

## Citation

If you use this work, please cite:

```bibtex
@misc{vit-rollout-pooling,
  title={Assessing the Beneficiality of Attention Rollout for Pooling in Vision Transformers},
  author={Shreejeet Sahay},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
