# Many-Shot Residual Stream Geometry

Investigating how transformer residual stream geometry evolves with context length during game-playing tasks.

## Key Findings

- **Tic-Tac-Toe shows dramatic dimensionality collapse** (participation ratio: 7.85 → 1.26) as games progress, driven by shrinking game state space — not a general many-shot learning effect
- **Stationary processes (Mess3 HMM, RPS) show flat geometry across context positions** — participation ratio stabilizes at ~2.0 within 5 tokens, matching the 2-simplex for 3-token vocabularies
- **Belief state regression quality decreases with context position** in Mess3 (R² from 1.0 → 0.95), suggesting representation drift in small transformers
- **The "many-shot simplification" hypothesis is not supported in its strong form** — geometric dimensionality tracks task structure (output distribution simplex), not amount of context
- **RPS counter strategy learns well** (val loss 0.554 vs entropy 1.099) without changing residual geometry relative to random opponents

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch numpy matplotlib scikit-learn scipy tqdm

# Run experiments (~10 min on GPU)
python src/run_experiments.py

# Additional visualizations
python src/extra_analysis.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review (pre-gathered)
├── resources.md                 # Resource catalog
├── src/
│   ├── model.py                 # SmallGPT decoder-only transformer
│   ├── train.py                 # Training loop
│   ├── analysis.py              # Geometric analysis functions
│   ├── run_experiments.py       # Main experiment runner
│   └── extra_analysis.py        # PCA scatter plots, heatmaps
├── results/
│   ├── experiment_results.json  # All numerical results
│   ├── models/                  # Saved model checkpoints
│   └── plots/                   # All visualizations
├── datasets/
│   ├── hmm/                     # HMM data generators
│   └── games/                   # Game data generators
├── papers/                      # Downloaded reference papers
└── code/                        # Cloned baseline repositories
```

See [REPORT.md](REPORT.md) for full analysis and discussion.
