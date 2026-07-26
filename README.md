# Semantic Vision Graphormer (SVG)

**Structured Inductive Biases via Spatial, Centrality, and Semantic Encodings for Vision Transformers**

*Master's Thesis Research | Tarbiat Modares University*  
*Author: Mohammad Mofidikhajeh | Advisor: Prof. Mansoor Rezghi*

## 📖 Overview
Standard Vision Transformers (ViTs) treat images as unstructured sequences of patches, forcing the model to implicitly learn spatial and structural relationships from massive amounts of data. **Semantic Vision Graphormer (SVG)** overcomes this "data hunger" by injecting explicit, lightweight graph-theoretic inductive biases directly into the self-attention mechanism. 

By formulating structural biases to be mathematically decoupled from Euclidean grid topologies, SVG not only achieves superior data efficiency in computer vision tasks but also establishes a parameterized framework for future extension to non-Euclidean manifolds (e.g., molecular graphs).

## ✨ Key Contributions
- **Explicit Structural Biases:** Injects spatial shortest-path distances, node centrality, and content-based semantic similarity directly into attention logits.
- **Parameter Efficiency:** Achieves a **4.01% accuracy improvement** over standard ViT baselines under constrained data regimes, adding **<15,000 parameters** and negligible MAC overhead.
- **Layer-Adaptive Scaling:** Introduces a dynamic scaling mechanism to modulate prior strength, allowing the model to hierarchically balance geometric and semantic reasoning across Transformer blocks.
- **Domain-Agnostic Formulation:** The core mathematical machinery is strictly decoupled from fixed lattices, providing a theoretical bridge to irregular, non-Euclidean graph representation learning.

## 🛠️ Requirements
The codebase is implemented in pure PyTorch.
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
