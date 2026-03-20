# Research Plan: Many-Shot Residual Stream Geometry

## Motivation & Novelty Assessment

### Why This Research Matters
Mechanistic interpretability aims to understand *how* transformers represent and process information internally. The residual stream is the central information highway, and understanding its geometry reveals how models encode beliefs and make decisions. In many-shot (long-context) settings, models narrow their output distributions — if this corresponds to geometric simplification in the residual stream, we gain a powerful tool for understanding and predicting model behavior.

### Gap in Existing Work
Shai et al. (2024) showed transformers encode belief state geometry in their residual streams, but only studied fixed-context HMM sequences without analyzing how geometry evolves *across context positions*. OthelloGPT work (Nanda et al., Li et al.) probed game state representations but didn't track geometric complexity as a function of context length. No work has:
1. Measured residual stream geometry simplification as context grows (many-shot effect)
2. Compared this across games of varying complexity
3. Connected many-shot distribution narrowing to geometric dimensionality reduction

### Our Novel Contribution
We test whether residual stream geometry measurably simplifies as transformers process more in-context examples in game-playing settings. We train small transformers on sequences from games of varying complexity (Mess3 HMM, Rock-Paper-Scissors with strategies, Tic-Tac-Toe) and measure:
- Effective dimensionality of residual stream activations vs context position
- Linear probe accuracy for latent state recovery vs context position
- Belief state regression quality vs context position

### Experiment Justification
- **Experiment 1 (Mess3 HMM baseline)**: Validates our framework against Shai et al.'s established results. Known ground-truth belief geometry allows precise measurement.
- **Experiment 2 (RPS with strategies)**: Tests whether structured opponent strategies create simpler geometry than random play, and whether geometry simplifies with more context.
- **Experiment 3 (Cross-game comparison)**: Compares geometric complexity across games (Mess3 vs RPS vs Tic-Tac-Toe) to test whether simpler games yield simpler geometry.

## Research Question
Does the residual stream geometry of transformers trained on game sequences simplify measurably as the model processes more in-context examples (many-shot effect), and can this simplification be characterized by reduced effective dimensionality and improved linear separability?

## Hypothesis Decomposition
1. **H1**: Effective dimensionality of residual stream activations decreases with context position (later positions → simpler geometry)
2. **H2**: Linear probe accuracy for latent game state increases with context position
3. **H3**: Games with more structured/predictable strategies show faster geometric simplification
4. **H4**: Simpler games (RPS) have lower-dimensional residual stream geometry than more complex games (Tic-Tac-Toe)

## Proposed Methodology

### Approach
Train small decoder-only transformers (GPT-style, 2-4 layers) on next-token prediction for game sequences. Extract residual stream activations at every layer and context position using hooks. Analyze geometry as a function of context position to detect many-shot simplification.

### Experimental Steps
1. Generate synthetic data: Mess3 HMM (10K sequences × 256 tokens), RPS with counter/random/mixed strategies (10K × 100 tokens each), Tic-Tac-Toe (10K games)
2. Train separate transformers for each data source (next-token prediction)
3. Extract residual stream activations at each layer and position
4. Compute geometric metrics: participation ratio (effective dimensionality), PCA explained variance, linear probe accuracy for latent states
5. Plot metrics vs context position to visualize many-shot effect
6. Compare across games and strategies

### Baselines
- Random (untrained) model activations
- Next-token probability baseline (does geometry encode more than next-token?)
- Shuffle control (break input-activation correspondence)

### Evaluation Metrics
- **Participation ratio** (effective dimensionality): PR = (Σλ_i)² / Σλ_i² where λ_i are eigenvalues of activation covariance
- **PCA explained variance**: Fraction of variance in top-k components vs context position
- **Linear probe accuracy**: Train linear classifier on activations → latent state; measure accuracy vs context position
- **R² of belief state regression**: For Mess3, regress activations onto ground-truth belief states

### Statistical Analysis Plan
- Bootstrap confidence intervals (95%) for all metrics
- Spearman correlation between context position and each metric
- Paired t-tests comparing early vs late context positions

## Expected Outcomes
- Effective dimensionality should decrease with context position (supporting H1)
- Probe accuracy should increase with context position (supporting H2)
- Structured strategies should show faster convergence (supporting H3)
- RPS geometry should be lower-dimensional than Tic-Tac-Toe (supporting H4)

## Timeline and Milestones
1. Environment setup + data generation: ~10 min
2. Model training (4 models × ~15 min each): ~60 min
3. Activation extraction + analysis: ~30 min
4. Visualization + documentation: ~30 min

## Potential Challenges
- Small transformers may not learn game structure well → use sufficient training
- RPS with random opponents may not show geometry simplification → compare with structured opponents
- Tic-Tac-Toe has variable-length games → use padding and masking

## Success Criteria
- Clear trend of dimensionality reduction with context position in at least one game
- Statistical significance (p < 0.05) for correlation between position and dimensionality
- Meaningful difference in geometry between structured and random games
