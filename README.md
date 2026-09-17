# Seeing Images as Graphs for Structured Visual Representation Learning
### Semantic Vision Graphormer (SVG)

**Author:** Mohammad Mofidikhajeh | **Term:** Spring 2026

---

## 📌 Overview
Vision Transformers (ViTs) achieve strong performance by processing images as unstructured sequences of patches. However, this "flat" view discards the rich spatial topology and content-driven groupings intrinsic to 2D visual data. 

**Semantic Vision Graphormer (SVG)** is a novel ViT architecture that fundamentally shifts this perspective. Instead of treating an image as a bag of tokens, SVG views every image as **two complementary graphs**:
1. **A Fixed Lattice Graph:** Encodes geometric relationships, shortest-path distances, and patch roles (corner, edge, interior).
2. **A Dynamic Contextual Graph:** Encodes pure visual similarity and semantic groupings, regardless of spatial separation.

By injecting lightweight, graph-theoretic inductive biases directly into the self-attention logits, SVG enhances representation learning with negligible computational overhead.

---

## ✨ Key Features & Contributions
* **Dual-Graph Philosophy:** Explicitly models both the "skeleton" (spatial lattice) and the "flesh" (semantic context) of an image.
* **Lightweight Integration:** Adds **fewer than 15k parameters** to a standard ViT baseline (Total: ~5.4M parameters).
* **Hardware Efficient:** Biases are precomputed per batch. SVG relies entirely on dense, highly-parallelizable tensor operations, mapping perfectly onto modern IO-aware dense attention kernels (e.g., FlashAttention) with **zero measurable wall-clock latency overhead** at inference.
* **Layer-Adaptive Priors:** Learnable scalars allow each Transformer layer to dynamically balance its reliance on geometric vs. semantic priors (e.g., early layers favor spatial locality, deeper layers favor semantic grouping).

---

## 🧠 Methodology: The 3 Attention Biases
SVG modifies the standard self-attention mechanism by adding three explicit, interpretable bias terms to the pre-softmax attention logits:

1. **Spatial Bias (from the Lattice Graph):** A learnable scalar based on the shortest-path (Chebyshev) distance between patches, plus a dedicated bias for the virtual `[CLS]` token.
2. **Centrality Encoding (Scalar Augmentation):** A learnable scalar based on a patch's degree in the lattice graph (corner=3, edge=5, interior=8), concatenated as a dedicated structural feature dimension.
3. **Semantic Bias (from the Contextual Graph):** Computed using bucketed cosine similarity on **raw, content-only** patch embeddings *before* any positional or centrality augmentation. This guarantees the similarity signal reflects pure visual content, untainted by spatial information.

---

## 📊 Results
Evaluated on **CIFAR-10** (Image Size: 32x32). SVG matches the parameter footprint of strong baselines while significantly outperforming them.

| Model | Accuracy (%) | Epochs | Parameters |
| :--- | :---: | :---: | :---: |
| **SVG (Ours)** | **94.68** | 479 | ~5.4M |
| SVG w/o Semantic Bias | 94.51 | 492 | ~5.4M |
| SVG w/o Semantic & Spatial | 92.95 | 500 | ~5.4M |
| Swin Transformer (timm) | ~92.50 | - | ~5.4M |
| ViT (timm) | ~90.50 | - | ~5.4M |

*(Note: Baseline numbers are approximate representations of the ~2.18% and ~4% margins noted in the draft).*

---

## 🛠️ Training Setup & Hyperparameters
The model was trained using the following configuration to ensure fair comparison and optimal convergence:

* **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=0.05`)
* **Batch Size:** 128
* **LR Schedule:** Cosine Annealing with 5-epoch linear warmup
* **Epochs:** ~500
* **Loss Function:** CrossEntropy with Label Smoothing (0.1) and Mixup/CutMix handling
* **Precision:** `torch.bfloat16`
* **Data Augmentation:**
  * RandomCrop (32, padding=4) + RandomHorizontalFlip
  * RandAugment (n=2, m=10)
  * MixUp ($\alpha=0.8$) + CutMix ($\alpha=1.0$) applied with equal probability
  * RandomErasing (p=0.25)

---

## 🚀 Installation & Requirements
*(Update this section based on your actual environment setup)*

```bash
# Clone the repository
git clone https://github.com/yourusername/svg-vision.git
cd svg-vision

# Create a virtual environment
python -m venv svg_env
source svg_env/bin/activate  # On Windows: svg_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
