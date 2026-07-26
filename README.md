# Semantic Vision Graphormer (SVG)
**Structured Inductive Biases via Spatial, Centrality, and Semantic Encodings for Vision Transformers**

## Overview
Semantic Vision Graphormer (SVG) is a novel Transformer framework that injects explicit, lightweight graph-theoretic inductive biases directly into self-attention mechanisms. By treating image patches as nodes in a dual-graph structure, SVG balances geometric and semantic reasoning hierarchically. The architecture is designed to overcome the "data hunger" of standard Vision Transformers (ViTs), achieving superior performance under constrained data regimes without heavy computational overhead.

## Key Highlights
* **Explicit Structural Priors:** Integrates spatial shortest-path, node centrality, and content-based semantic similarity directly into attention logits.
* **High Parameter Efficiency:** Adds **<15,000 parameters** and negligible MAC overhead compared to standard ViT baselines.
* **Constrained Data Performance:** Achieves a **4.01% accuracy improvement** over baseline ViTs in low-data regimes.
* **Layer-Adaptive Scaling:** Dynamically modulates prior strength across network depth, allowing the model to hierarchically balance geometric and semantic reasoning.
* **Domain-Agnostic Mathematics:** While validated on Euclidean vision grids, the core mathematical formulation of the dual-graph bias injection is decoupled from grid topologies, providing a foundation for future extension to non-Euclidean manifolds (e.g., molecular or chemical graphs).
