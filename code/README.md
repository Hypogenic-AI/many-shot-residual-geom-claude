# Many-Shot Residual Stream Geometry -- Code Repositories

This directory contains cloned repositories relevant to the research project on
how transformers represent belief state geometry in their residual streams,
particularly in game-playing contexts.

---

## 1. BeliefStateTransformer
- **Source:** https://github.com/sanowl/BeliefStateTransformer
- **Description:** Community implementation extending experiments from Shai et al.
  (2024) "Transformers Represent Belief State Geometry in their Residual Stream"
  (NeurIPS 2024, arXiv:2405.15943). Provides a framework for analyzing how
  transformer models encode belief states when trained on data from Hidden Markov
  Models (HMMs).
- **Key functionality:**
  - Data generation for Mess3 and RRXOR processes
  - Simple transformer architecture matching the paper
  - Belief state analysis in the residual stream via PCA-based 3D plots
  - Layer-wise analysis of representations across network depth
  - Training dynamics tracking (belief state evolution during training)
  - Mixed State Presentation (MSP) structure visualization
- **Note:** This is NOT the official author repo (no official code repo was found
  for the Shai et al. 2024 paper). This is a third-party reimplementation.

## 2. Fractal-MSPs
- **Source:** https://github.com/wz-ml/Fractal-MSPs
- **Description:** Another reimplementation of Adam Shai's "Transformers Represent
  Belief State Geometry in their Residual Stream." Focuses on reproducing the
  fractal-like Mixed State Presentation structures.
- **Key functionality:**
  - Reimplementation of belief state geometry experiments
  - Fractal MSP visualization

## 3. markov-transformers
- **Source:** https://github.com/lena-lenkeit/markov-transformers
- **Description:** Trains small transformers on HMM-generated data and attempts to
  extract the full HMM (transition and emission matrices) plus belief states
  directly from the transformer residual stream -- without knowing the original
  HMM.
- **Key functionality:**
  - Unsupervised HMM decoding from transformer residual streams
  - Inference of transition/emission matrices from activations
  - Works on the Mess3 process with 1-layer decoder-only transformers
  - Inspired directly by the Shai et al. belief state geometry findings

## 4. mech_int_othelloGPT
- **Source:** https://github.com/ajyl/mech_int_othelloGPT
- **Paper:** "Emergent Linear Representations in World Models of Self-Supervised
  Sequence Models" by Neel Nanda and Andrew Lee
- **Description:** Mechanistic interpretability analysis of OthelloGPT showing that
  Othello-playing neural networks learn linear representations of board state.
  Demonstrates that probing for "my colour" vs "opponent's colour" is a powerful
  way to interpret internal state, and that linear representations enable
  controlling model behaviour with vector arithmetic.
- **Key functionality:**
  - Linear board state probes (`board_probe.py`)
  - "Flipped" probes for alternative representations (`train_flipped.py`)
  - Intervention experiments (`intervene.py`, `intervene_blank.py`, `intervene_flipped.py`)
  - Utility functions for TransformerLens-based analysis (`tl_othello_utils.py`)
  - Figure generation notebooks

## 5. othello_world
- **Source:** https://github.com/likenneth/othello_world
- **Paper:** "Emergent World Representations: Exploring a Sequence Model Trained on
  a Synthetic Task" (ICLR 2023) by Kenneth Li et al.
- **Description:** The original Othello-GPT project. Trains a GPT variant to predict
  legal Othello moves and uncovers emergent nonlinear internal representations of
  board state. Interventional experiments show these representations can control
  network output.
- **Key functionality:**
  - GPT training on championship and synthetic Othello datasets (based on minGPT)
  - Nonlinear and linear probe training (`train_probe_othello.py`)
  - Intervention experiments (`intervening_probe_interact_column.ipynb`)
  - Attribution via intervention plots
  - Pre-trained checkpoints available via Google Drive

## 6. TransformerLens
- **Source:** https://github.com/TransformerLensOrg/TransformerLens
- **Created by:** Neel Nanda
- **Description:** Library for mechanistic interpretability of GPT-2 style language
  models. Loads 50+ open source language models and exposes internal activations.
  Supports caching any internal activation and adding hook functions to edit,
  remove, or replace activations.
- **Key functionality:**
  - `HookedTransformer` class for loading and running models with activation access
  - Activation caching via `run_with_cache()`
  - Hook-based intervention system for editing activations mid-forward-pass
  - Support for GPT-2, GPT-Neo, OPT, Pythia, LLaMA, and many other architectures
  - Includes Othello-GPT demo notebook
  - Used extensively in published mech interp research (grokking, sparse probing,
    circuit discovery, etc.)

## 7. BST (Belief State Transformer)
- **Source:** https://github.com/microsoft/BST
- **Paper:** "The Belief State Transformer" (ICLR 2025) by Edward S. Hu, Kwangjun
  Ahn, Qinghua Liu, et al. (Microsoft Research)
- **Description:** Official codebase for the Belief State Transformer paper. Based on
  nanoGPT. Trains transformers that explicitly model belief states, addressing
  pitfalls of next-token prediction.
- **Key functionality:**
  - Star graph experiments (following Bachmann & Nagarajan "Pitfalls of Next-Token
    Prediction" setup)
  - Data preparation for various graph configurations (G(2,5), G(2,10), G(5,5))
  - BST model training with YAML-based configuration
  - Requires PyTorch 2.6+

---

## Repositories NOT Found

### Shai et al. 2024 -- Official Code
- **Paper:** "Transformers represent belief state geometry in their residual stream"
  (arXiv:2405.15943, NeurIPS 2024)
- **Authors:** Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink
  Oldenziel, Paul M. Riechers
- **Status:** No official code repository was found. The community reimplementations
  (BeliefStateTransformer, Fractal-MSPs) cover the key experiments.

### Piotrowski et al. 2025 -- Constrained Belief Updates
- **Paper:** "Constrained belief updates explain geometric structures in transformer
  representations" (arXiv:2502.01954)
- **Authors:** Mateusz Piotrowski, Paul M. Riechers, Daniel Filan, Adam S. Shai
- **Status:** No public code repository was found as of March 2026. This paper
  extends the Shai et al. 2024 work by showing transformers implement constrained
  Bayesian belief updating -- a parallelized version of partial Bayesian inference
  shaped by architectural constraints. It provides theoretical predictions for
  attention patterns, OV-vectors, and embedding vectors.

---

## Relevance Map

| Research Question | Primary Repos |
|---|---|
| Belief state geometry in residual streams | BeliefStateTransformer, Fractal-MSPs, markov-transformers |
| Constrained belief updates / computational mechanics | (no code yet -- see Piotrowski et al. 2025) |
| Othello world models & board state representations | othello_world, mech_int_othelloGPT |
| General mechanistic interpretability tooling | TransformerLens |
| Belief state transformers (explicit belief modeling) | BST |
| HMM extraction from transformer activations | markov-transformers |
