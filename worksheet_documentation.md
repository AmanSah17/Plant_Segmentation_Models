# PDLC-ViT: Plant Disease Localization and Classification Vision Transformer
*An Open-Source Implementation for Smart Agricultural Robotics and AI-Driven Organic Production*

---

## 1. Project Overview

This repository hosts a production-grade, open-source implementation of the **PDLC-ViT** (Plant Disease Localization and Classification Vision Transformer), a Multi-Task Learning (MTL) architecture designed for precision agriculture. The primary objective of this project is to provide a rigorous, highly optimized deep learning pipeline that simultaneously identifies the type of crop disease (classification) and precisely delineates its spatial footprint on the leaf (segmentation).

By open-sourcing this implementation, we aim to bridge the gap between state-of-the-art vision transformer research and real-world agricultural deployment, paving the way for cleaner, organic production pipelines monitored by smart robotics.

---

## 2. Advantages of the PDLC-ViT Approach

Traditional Convolutional Neural Networks (CNNs) often struggle to capture long-range global dependencies in complex foliar textures. The PDLC-ViT architecture overcomes this through a purely attention-based mechanism, offering distinct technical advantages:

1. **Joint Optimization (MTL):** By sharing a foundational representation space, the classification and segmentation tasks mutually regularize each other. The global context aids in accurate disease classification, while the classification signal enforces structural awareness in the segmentation mask.
2. **Multi-Scale Feature Fusion:** The co-scale mechanism aggregates patch tokens at varying granularities ($1\times1$, $2\times2$, $4\times4$), ensuring the network remains invariant to the scale of the disease manifestations (from tiny necrotic spots to widespread blight).
3. **Cross-Attention Representation:** Decoupling the initial feature extraction into parallel branches prevents task interference (negative transfer) before fusing the optimized representations via cross-attention.

---

## 3. Core Architecture Implementation

Our implementation strictly adheres to the mathematical formulations presented in the original PDLC-ViT paper, while engineering it for scalable, GPU-efficient execution.

### 3.1. Shared Patch Embedding
The input tensor $X \in \mathbb{R}^{H \times W \times C}$ is partitioned into non-overlapping patches $P \in \mathbb{R}^{N \times (P^2 \cdot C)}$, which are linearly projected into an embedding space of dimension $d$ (specifically, $d=768$). A learnable positional embedding is injected to retain spatial structural topology.

### 3.2. Parallel Co-Scale and Co-Attention Branches
To prevent gradient conflict between the localization and classification objectives early in the network, the embedded patches bifurcate into two task-specific streams:
* **Co-Scale Layer:** Applies Adaptive Average Pooling at dynamic grid resolutions to force the network to digest macro-level semantic concepts before projecting them back to the sequence length $N$.
* **Co-Attention Layer:** A Multi-Head Self-Attention (MHSA) block applied exclusively to the co-scaled tokens to refine inter-patch relationships independently for both tasks.

### 3.3. Cross-Attention Fusion Module
The representations from the independent branches ($Z_{seg}$ and $Z_{cls}$) are fused to synthesize a unified representation. The classification branch representation serves as the Query ($Q$), while the segmentation branch provides the Keys ($K$) and Values ($V$), calculating:
$$ \text{CrossAttn}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

### 3.4. Deep Transformer Encoder
The fused tokens are passed through a highly parameterized stack of Transformer blocks ($L=8$). Each block sequentially applies Layer Normalization, Multi-Head Self Attention (MHSA), and a Multi-Layer Perceptron (MLP) with GELU activations. 

### 3.5. Task-Specific Heads
* **Segmentation Head:** Reshapes the 1D token sequence back to a 2D spatial grid. It employs cascaded transpose convolutions to upsample the features, yielding a pixel-wise probability distribution $Y_{seg} \in \mathbb{R}^{115 \times H \times W}$.
* **Classification Head:** Fuses the global average pool of the shared representation $Z$, a pooled representation of the predicted segmentation mask, and a learned class embedding via Multi-Head Cross Attention (MHCA) to predict the final categorical probability $Y_{cls} \in \mathbb{R}^{114}$.

---

## 4. Our Engineering Contributions & Enhancements

While honoring the paper's architecture, our implementation introduces critical engineering enhancements to achieve production-level stability and accuracy:

1. **Massive Capacity Scaling:** We expanded the representation capacity significantly, operating with `EMBED_DIM=768`, `NUM_HEADS=8`, and `NUM_ENC_LAYERS=8`, yielding an ~86 Million parameter model.
2. **VRAM Optimization:** To allow this massive architecture to train on consumer GPUs (4GB VRAM), we implemented **Automatic Mixed Precision (AMP)** and decoupled the virtual batch size via **Gradient Accumulation** (Real Batch: 2, Accumulation Steps: 8).
3. **Data Leakage Eradication:** We discovered and eliminated a severe data-leakage vulnerability in the evaluation logic where ground-truth labels and masks were artificially inflating validation queries. Our validation loop now rigorously enforces strict inference-only query generation.
4. **Stratified K-Fold & Optuna MLflow Pipeline:** We developed a fully automated hyperparameter tuning suite using Optuna, integrated with a Stratified 5-Fold Cross-Validation pipeline. All metrics, learning rates, losses, and early-stopping triggers are logged directly to a local MLflow tracking server.

---

## 5. Future Roadmap: Precision Agriculture Web Ecosystem

This PyTorch implementation serves as the intelligence layer for a much broader initiative. Our future roadmap focuses on deploying this model into a **Full-Stack Web Application** designed for smart agricultural ecosystems.

### Phase 1: Interactive Monitoring Dashboard
* **Real-time Inference:** A robust backend (FastAPI/Django) serving the PDLC-ViT model via ONNX/TensorRT for low-latency image processing.
* **Diagnostic Reporting:** Users upload drone or smartphone imagery and receive instantaneous visual heatmaps of segmented disease locations alongside highly accurate classification.

### Phase 2: Actionable Remedy & Methodology Engine
* **Organic Production Methodologies:** Connecting identified diseases to a dynamic database of organic, chemical-free remedies.
* **Step-by-Step Clean Agriculture:** Generating customized, actionable workflows for farmers to isolate diseased crops, adjust local soil pH/humidity, and deploy biological controls rather than broad-spectrum pesticides.

### Phase 3: Smart Robotics Integration
* **Autonomous Drones & Rovers:** Extending the ONNX-exported model to edge devices (e.g., NVIDIA Jetson) mounted on agricultural rovers.
* **Surgical Interventions:** Guiding robotic arms using the precise segmentation masks to autonomously prune necrotic leaves or apply localized, micro-dose organic treatments, drastically reducing labor and chemical runoff.

---
*Documented with Technical Rigor for the Open-Source Community.*
