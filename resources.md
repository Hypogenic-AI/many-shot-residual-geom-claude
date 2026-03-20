# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Many-Shot Residual Stream Geometry" research project. The project investigates how residual stream geometry simplifies in transformers during many-shot learning, particularly when playing simple games.

---

## Papers
Total papers downloaded: 22

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Transformers Represent Belief State Geometry in their Residual Stream | Shai, Marzen, Teixeira, Oldenziel, Riechers | 2024 | papers/li_2024_belief_state_geometry.pdf | **CORE** — Belief states linearly encoded in residual stream |
| 2 | Emergent Linear Representations in World Models (OthelloGPT) | Nanda, Lee, Wattenberg | 2023 | papers/nanda_2023_emergent_linear_representations.pdf | **CORE** — Linear world model in game-playing transformer |
| 3 | Constrained Belief Updates Explain Geometric Structures | Piotrowski, Riechers, Filan, Shai | 2025 | papers/constrained_belief_updates_geometry.pdf | **CORE** — How architectural constraints shape belief geometry |
| 4 | Not All Language Model Features Are One-Dimensionally Linear | Engels et al. | 2024 | papers/not_all_features_linear.pdf | Multi-dimensional features in residual stream |
| 5 | Residual Stream Analysis with Multi-Layer SAEs | Various | 2024 | papers/residual_stream_multi_layer_sae.pdf | Multi-layer residual stream analysis |
| 6 | Curved Inference: Concern-Sensitive Geometry in LLM Residual Streams | Various | 2025 | papers/curved_inference_geometry.pdf | Non-Euclidean geometry of residual streams |
| 7 | The Geometry of Thought | Various | 2026 | papers/geometry_of_thought.pdf | Scale effects on reasoning geometry |
| 8 | The Bayesian Geometry of Transformer Attention | Various | 2025 | papers/bayesian_geometry_attention.pdf | Formal Bayesian geometry of attention |
| 9 | Geometric Scaling of Bayesian Inference in LLMs | Various | 2025 | papers/geometric_scaling_bayesian.pdf | Scaling of Bayesian geometry |
| 10 | The Geometry of BERT | Various | 2025 | papers/geometry_bert.pdf | Geometric analysis of BERT |
| 11 | Tracr: Compiled Transformers | Lindner et al. | 2023 | papers/tracr_compiled_transformers.pdf | Compiled transformers for interpretability |
| 12 | Talking Heads: Inter-layer Communication | Various | 2024 | papers/talking_heads_inter_layer.pdf | Information flow between layers |
| 13 | Is This the Subspace You Are Looking for? | Various | 2023 | papers/subspace_interpretability_illusion.pdf | Interpretability illusion warning |
| 14 | Dictionary Learning in Othello-GPT | He et al. | 2024 | papers/dictionary_learning_othello.pdf | SAE circuit discovery in game transformer |
| 15 | In-Context Learning and Induction Heads | Olsson et al. | 2022 | papers/olsson_2022_induction_heads.pdf | ICL mechanism via induction heads |
| 16 | What Can Transformers Learn In-Context? | Garg et al. | 2022 | papers/garg_2022_icl_function_classes.pdf | ICL function classes |
| 17 | What Learning Algorithm is ICL? | Akyurek et al. | 2023 | papers/akyurek_2023_icl_linear_models.pdf | ICL as implicit learning |
| 18 | Othello-GPT (Original) | Li et al. | 2023 | papers/li_2023_othello_gpt.pdf | Original Othello world model discovery |
| 19 | Controllable Context Sensitivity | Various | 2024 | papers/context_sensitivity_knob.pdf | Context vs prior control mechanism |
| 20 | Interpreting Factual Recall Mechanisms | Various | 2024 | papers/interpreting_factual_recall.pdf | Factual recall in transformers |
| 21 | Clustering in Hardmax Transformers | Various | 2024 | papers/clustering_hardmax_transformers.pdf | Clustering dynamics theory |
| 22 | Transformer Dynamics: Neuroscientific Approach | Various | 2025 | papers/transformer_dynamics_neuroscience.pdf | Neuroscience-inspired interpretability |

See papers/README.md for detailed descriptions.

---

## Datasets
Total dataset generators: 3 (covering 7 processes/games)

| Name | Source | Type | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Mess3 HMM | Custom generator | Synthetic | Sequence prediction | datasets/hmm/ | 3-state HMM, fractal belief geometry |
| Z1R HMM | Custom generator | Synthetic | Sequence prediction | datasets/hmm/ | ...01R01R... pattern |
| RRXOR HMM | Custom generator | Synthetic | Sequence prediction | datasets/hmm/ | Random-Random-XOR, 36 belief states |
| Rock-Paper-Scissors | Custom generator | Synthetic | Game sequences | datasets/games/ | Multiple opponent strategies |
| Tic-Tac-Toe | Custom generator | Synthetic | Game transcripts | datasets/games/ | Random and heuristic play |
| Kuhn Poker | Custom generator | Synthetic | Game hands | datasets/games/ | 3-card poker, Nash equilibrium known |
| Othello transcripts | othello_world repo | Synthetic | Legal move sequences | datasets/othello/ | 60-token vocabulary |

See datasets/README.md for generation instructions and detailed descriptions.

---

## Code Repositories
Total repositories cloned: 7

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| BeliefStateTransformer | github.com/sanowl/BeliefStateTransformer | Belief state geometry experiments | code/BeliefStateTransformer/ | Community reimplementation of Shai et al. 2024 |
| Fractal-MSPs | github.com/wz-ml/Fractal-MSPs | Fractal MSP visualization | code/Fractal-MSPs/ | Another Shai et al. reimplementation |
| markov-transformers | github.com/lena-lenkeit/markov-transformers | HMM extraction from transformers | code/markov-transformers/ | Extract HMMs from residual streams |
| mech_int_othelloGPT | github.com/ajyl/mech_int_othelloGPT | Linear board state probes | code/mech_int_othelloGPT/ | Nanda & Lee OthelloGPT analysis |
| othello_world | github.com/likenneth/othello_world | Original OthelloGPT | code/othello_world/ | Li et al. 2023, includes data generation |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability library | code/TransformerLens/ | Neel Nanda's mech interp toolkit |
| BST | github.com/microsoft/BST | Belief State Transformer | code/BST/ | Microsoft Research, ICLR 2025 |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for "residual stream geometry transformer mechanistic interpretability"
2. Additional targeted searches for "many-shot in-context learning" and "game-playing transformer interpretability"
3. GitHub searches for "belief state geometry transformer", "computational mechanics transformer", "residual stream geometry"
4. ArXiv API and Semantic Scholar for paper metadata and arxiv IDs

### Selection Criteria
- Papers with relevance score ≥ 2 from paper-finder (50 papers met threshold, 22 downloaded)
- Prioritized: (1) residual stream geometry, (2) belief state / computational mechanics, (3) game-playing interpretability, (4) in-context learning mechanisms
- Code repos: prioritized official implementations and repos directly implementing belief state analysis

### Challenges Encountered
- No official code repository found for Shai et al. 2024 (core paper) — used community reimplementations
- No official code for Piotrowski et al. 2025 (constrained belief updates)
- Semantic Scholar API rate limiting required switching to arxiv API for paper metadata
- One paper download (constrained_belief_updates_geometry) initially had wrong arxiv ID — corrected

### Gaps and Workarounds
- **No existing many-shot analysis code**: Created custom game data generators (RPS, tic-tac-toe, poker) since no existing datasets specifically target many-shot residual stream analysis
- **No simple game transformer models**: Will need to train from scratch; TransformerLens and BeliefStateTransformer repos provide the necessary infrastructure
- **HMM data is synthetic**: This is actually ideal for controlled experiments with known ground truth

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Start with Mess3 HMM** — reproduces Shai et al. results, validates framework
- **Then RPS/Tic-Tac-Toe** — simple games with analytically tractable state spaces
- **Then Othello** — more complex game, can leverage existing OthelloGPT infrastructure

### 2. Baseline Methods
- Linear probing of residual stream (Shai et al. framework)
- PCA/dimensionality analysis at varying context positions
- Random model baseline
- Next-token prediction baseline (test if geometry goes beyond next-token)

### 3. Evaluation Metrics
- MSE of belief state regression vs context position (many-shot curve)
- Effective dimensionality of residual stream activations vs context length
- R² of pairwise distance preservation
- Probe accuracy as function of in-context examples seen

### 4. Code to Adapt/Reuse
- **BeliefStateTransformer** — most directly applicable, has data generation + analysis pipeline
- **TransformerLens** — for detailed mechanistic analysis (hook-based activation access)
- **mech_int_othelloGPT** — for Othello-specific experiments
- **datasets/hmm/generate_hmm_data.py** — GenericHMM class with belief_update() for computing ground truth
- **datasets/games/generate_game_data.py** — game data generation for new experiments
