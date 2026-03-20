"""
Main experiment runner: Many-Shot Residual Stream Geometry.

Trains transformers on game/HMM sequences and analyzes how residual stream
geometry evolves across context positions (many-shot effect).
"""
import sys
import json
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "hmm"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "games"))

from model import SmallGPT
from train import train_model
from analysis import (
    participation_ratio, pca_explained_variance, analyze_geometry_by_position,
    analyze_probe_by_position, belief_state_regression_r2, pairwise_distance_correlation
)
from generate_hmm_data import Mess3, GenericHMM, generate_dataset
from generate_game_data import RPSGenerator, TicTacToeGenerator


# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================
MODEL_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 4,
    "epochs": 60,
    "lr": 3e-4,
    "batch_size": 128,
}

NUM_SEQUENCES = 10000
SEQ_LENGTH = 128  # Context length for analysis
NUM_EXTRACT = 2000  # Number of sequences for activation extraction


# ============================================================================
# Data Generation
# ============================================================================
def generate_all_data():
    """Generate all datasets."""
    print("=" * 60)
    print("GENERATING DATA")
    print("=" * 60)

    datasets = {}

    # 1. Mess3 HMM (known ground truth)
    print("\n[1] Mess3 HMM...")
    mess3 = Mess3(emission_prob=0.9)
    mess3_generic = GenericHMM(mess3.transition, mess3.emission, mess3.initial, "mess3")
    data = generate_dataset("mess3", NUM_SEQUENCES, SEQ_LENGTH, seed=SEED)
    datasets["mess3"] = {
        "emissions": data["emissions"],
        "states": data["states"],
        "vocab_size": 3,
        "hmm": mess3_generic,
        "description": "Mess3 HMM (3 states, fractal geometry)"
    }
    print(f"  Shape: {data['emissions'].shape}, vocab: 3")

    # 2. RPS - counter strategy (structured, predictable)
    print("\n[2] RPS (counter strategy)...")
    rps_counter = RPSGenerator.generate_dataset(NUM_SEQUENCES, rounds_per_game=SEQ_LENGTH // 2,
                                                 strategy="counter", seed=SEED)
    datasets["rps_counter"] = {
        "emissions": rps_counter,
        "states": None,  # No hidden states for RPS
        "vocab_size": 3,
        "description": "RPS with counter strategy"
    }
    print(f"  Shape: {rps_counter.shape}")

    # 3. RPS - random strategy (unstructured)
    print("\n[3] RPS (random strategy)...")
    rps_random = RPSGenerator.generate_dataset(NUM_SEQUENCES, rounds_per_game=SEQ_LENGTH // 2,
                                                strategy="random", seed=SEED)
    datasets["rps_random"] = {
        "emissions": rps_random,
        "states": None,
        "vocab_size": 3,
        "description": "RPS with random strategy"
    }
    print(f"  Shape: {rps_random.shape}")

    # 4. RPS - win-stay-lose-shift (intermediate structure)
    print("\n[4] RPS (WSLS strategy)...")
    rps_wsls = RPSGenerator.generate_dataset(NUM_SEQUENCES, rounds_per_game=SEQ_LENGTH // 2,
                                              strategy="win_stay_lose_shift", seed=SEED)
    datasets["rps_wsls"] = {
        "emissions": rps_wsls,
        "states": None,
        "vocab_size": 3,
        "description": "RPS with win-stay-lose-shift"
    }
    print(f"  Shape: {rps_wsls.shape}")

    # 5. Tic-Tac-Toe (more complex game)
    print("\n[5] Tic-Tac-Toe...")
    ttt_data, ttt_outcomes = TicTacToeGenerator.generate_dataset(NUM_SEQUENCES, strategy="random", seed=SEED)
    # For TTT, we need to handle variable length games. Pad with vocab_size token.
    ttt_vocab = 10  # 0-8 positions + 9 separator
    # Replace -1 padding with separator token (9)
    ttt_padded = ttt_data.copy()
    ttt_padded[ttt_padded == -1] = 9
    datasets["tictactoe"] = {
        "emissions": ttt_padded,
        "states": None,
        "vocab_size": ttt_vocab,
        "outcomes": ttt_outcomes,
        "description": "Tic-Tac-Toe (random play)"
    }
    print(f"  Shape: {ttt_padded.shape}, vocab: {ttt_vocab}")

    # 6. Mess3 with weaker emission (more ambiguous)
    print("\n[6] Mess3 (weak emission p=0.6)...")
    mess3_weak = Mess3(emission_prob=0.6)
    mess3_weak_generic = GenericHMM(mess3_weak.transition, mess3_weak.emission, mess3_weak.initial, "mess3_weak")
    data_weak = generate_dataset("mess3", NUM_SEQUENCES, SEQ_LENGTH, seed=SEED + 1)
    # Regenerate with weak emission
    rng = np.random.default_rng(SEED + 1)
    weak_states = np.zeros((NUM_SEQUENCES, SEQ_LENGTH), dtype=np.int32)
    weak_emissions = np.zeros((NUM_SEQUENCES, SEQ_LENGTH), dtype=np.int32)
    for i in range(NUM_SEQUENCES):
        s, e = mess3_weak.generate(SEQ_LENGTH, rng)
        weak_states[i] = s
        weak_emissions[i] = e
    datasets["mess3_weak"] = {
        "emissions": weak_emissions,
        "states": weak_states,
        "vocab_size": 3,
        "hmm": mess3_weak_generic,
        "description": "Mess3 HMM (weak emission p=0.6)"
    }
    print(f"  Shape: {weak_emissions.shape}")

    return datasets


# ============================================================================
# Training
# ============================================================================
def train_all_models(datasets):
    """Train a transformer for each dataset."""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    models = {}
    histories = {}

    for name, ds in datasets.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()

        # For TTT, sequences are short (max 9), need different config
        config = MODEL_CONFIG.copy()
        if name == "tictactoe":
            config["epochs"] = 100  # More epochs for short sequences
            config["d_model"] = 32  # Smaller model for short sequences
            config["n_layers"] = 2

        model, hist = train_model(
            name, ds["emissions"], ds["vocab_size"], config,
            device=DEVICE, save_dir=str(MODELS_DIR)
        )
        models[name] = model
        histories[name] = hist
        print(f"  Time: {time.time() - t0:.1f}s")

    return models, histories


# ============================================================================
# Activation Extraction
# ============================================================================
def extract_activations(models, datasets, n_samples=NUM_EXTRACT):
    """Extract residual stream activations from all models."""
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)

    all_streams = {}
    for name, model in models.items():
        print(f"\n  Extracting {name}...")
        model.eval()
        ds = datasets[name]
        data = ds["emissions"][:n_samples]

        # For decoder model, input is tokens[:-1]
        X = torch.tensor(data[:, :-1], dtype=torch.long).to(DEVICE)

        # Extract in batches
        batch_size = 256
        all_layer_streams = None
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            streams = model.get_all_residual_streams(batch)
            if all_layer_streams is None:
                all_layer_streams = {k: [v] for k, v in streams.items()}
            else:
                for k, v in streams.items():
                    all_layer_streams[k].append(v)

        # Concatenate batches
        for k in all_layer_streams:
            all_layer_streams[k] = torch.cat(all_layer_streams[k], dim=0).numpy()

        all_streams[name] = all_layer_streams
        n_layers = model.n_layers
        T = all_layer_streams[0].shape[1]
        print(f"    Layers: {n_layers + 2}, seq_len: {T}, n_samples: {len(data)}")

    return all_streams


# ============================================================================
# Geometry Analysis
# ============================================================================
def analyze_all(all_streams, datasets, models):
    """Run geometry analysis on all models."""
    print("\n" + "=" * 60)
    print("ANALYZING GEOMETRY")
    print("=" * 60)

    results = {}

    for name, streams in all_streams.items():
        print(f"\n--- Analyzing {name} ---")
        ds = datasets[name]
        model = models[name]
        n_layers = model.n_layers

        # Use final layer (after last block, before LN)
        final_layer = n_layers  # Last transformer block output

        # 1. Geometry by position (participation ratio, PCA)
        print("  Computing geometry by position...")
        geom = analyze_geometry_by_position(streams, final_layer)
        positions = sorted(geom.keys())
        pr_values = [geom[p]["participation_ratio"] for p in positions]
        pca3_values = [geom[p]["pca_var_3"] for p in positions]
        pca5_values = [geom[p]["pca_var_5"] for p in positions]

        # 2. Multi-layer analysis (compare early vs late layers)
        print("  Computing multi-layer geometry...")
        layer_pr = {}
        for layer_idx in range(n_layers + 2):  # 0 to n_layers+1
            # Average PR across all positions
            acts = streams[layer_idx]  # (N, T, d)
            prs = []
            for pos in range(0, acts.shape[1], max(1, acts.shape[1] // 10)):
                prs.append(participation_ratio(acts[:, pos, :]))
            layer_pr[layer_idx] = np.mean(prs)

        # 3. Belief state regression (for Mess3 datasets)
        belief_r2_by_pos = {}
        dist_corr_by_pos = {}
        if "hmm" in ds and ds["states"] is not None:
            print("  Computing belief state regression...")
            hmm = ds["hmm"]
            emissions = ds["emissions"][:NUM_EXTRACT]
            states = ds["states"][:NUM_EXTRACT]

            # Compute belief trajectories
            n_samp = min(500, len(emissions))  # Limit for speed
            belief_trajs = np.zeros((n_samp, emissions.shape[1], hmm.num_states))
            for i in range(n_samp):
                beliefs = hmm.compute_belief_trajectory(emissions[i])
                belief_trajs[i] = beliefs[1:]  # Skip prior, align with positions

            acts = streams[final_layer][:n_samp]  # (n_samp, T, d)

            for pos in positions:
                if pos >= belief_trajs.shape[1]:
                    continue
                a = acts[:, pos, :]
                b = belief_trajs[:, pos, :]
                r2 = belief_state_regression_r2(a, b)
                belief_r2_by_pos[pos] = r2

                if len(a) >= 50:
                    dc = pairwise_distance_correlation(a, b)
                    dist_corr_by_pos[pos] = dc

        # 4. Probe for game-specific labels
        probe_by_pos = {}
        if name.startswith("rps"):
            # For RPS, probe for "who won the last round"
            print("  Computing probe for RPS outcome...")
            data = ds["emissions"][:NUM_EXTRACT, :-1]  # Align with model input
            # Label: outcome of most recent complete round (win/lose/draw for p2)
            # Rounds are pairs (p1, p2). At position t (0-indexed), if t is odd (p2 just played),
            # we can determine outcome.
            labels = np.zeros(data.shape, dtype=np.int32)
            BEATS = {0: 2, 1: 0, 2: 1}
            for i in range(len(data)):
                for t in range(1, data.shape[1], 2):
                    p1, p2 = data[i, t-1], data[i, t]
                    if BEATS[p2] == p1:
                        labels[i, t] = 2  # p2 wins
                    elif BEATS[p1] == p2:
                        labels[i, t] = 0  # p1 wins
                    else:
                        labels[i, t] = 1  # draw

            # Probe at odd positions only (after p2 plays)
            odd_positions = [p for p in positions if p % 2 == 1 and p < data.shape[1]]
            if odd_positions:
                probe_by_pos = analyze_probe_by_position(
                    streams, final_layer, labels, odd_positions
                )

        elif ds.get("states") is not None:
            # For HMM, probe for hidden state
            print("  Computing probe for hidden state...")
            states_aligned = ds["states"][:NUM_EXTRACT, :-1]
            probe_by_pos = analyze_probe_by_position(
                streams, final_layer, states_aligned, positions
            )

        results[name] = {
            "positions": positions,
            "participation_ratio": pr_values,
            "pca_var_3": pca3_values,
            "pca_var_5": pca5_values,
            "layer_pr": layer_pr,
            "belief_r2_by_pos": belief_r2_by_pos,
            "dist_corr_by_pos": dist_corr_by_pos,
            "probe_by_pos": {int(k): v for k, v in probe_by_pos.items()},
        }

        # Print summary
        print(f"  PR: {pr_values[0]:.2f} (pos 0) -> {pr_values[-1]:.2f} (pos {positions[-1]})")
        if belief_r2_by_pos:
            r2_vals = list(belief_r2_by_pos.values())
            print(f"  Belief R²: {r2_vals[0]:.3f} (early) -> {r2_vals[-1]:.3f} (late)")
        if probe_by_pos:
            probe_vals = list(probe_by_pos.values())
            print(f"  Probe acc: {probe_vals[0][0]:.3f} -> {probe_vals[-1][0]:.3f}")

    return results


# ============================================================================
# Visualization
# ============================================================================
def plot_results(results, histories):
    """Create all plots."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Color scheme
    colors = {
        "mess3": "#1f77b4",
        "mess3_weak": "#aec7e8",
        "rps_counter": "#ff7f0e",
        "rps_random": "#2ca02c",
        "rps_wsls": "#d62728",
        "tictactoe": "#9467bd",
    }

    # ---- Plot 1: Participation Ratio vs Position (main result) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        if name == "tictactoe":
            continue  # Different seq length, plot separately
        positions = res["positions"]
        pr = res["participation_ratio"]
        ax.plot(positions, pr, '-o', label=name, color=colors.get(name, "gray"),
                markersize=3, linewidth=1.5)
    ax.set_xlabel("Context Position", fontsize=12)
    ax.set_ylabel("Participation Ratio (Effective Dimensionality)", fontsize=12)
    ax.set_title("Residual Stream Dimensionality vs Context Position", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "participation_ratio_vs_position.png", dpi=150)
    plt.close()
    print("  Saved: participation_ratio_vs_position.png")

    # ---- Plot 2: PCA Explained Variance (top 3 components) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        if name == "tictactoe":
            continue
        positions = res["positions"]
        pca3 = res["pca_var_3"]
        ax.plot(positions, pca3, '-o', label=name, color=colors.get(name, "gray"),
                markersize=3, linewidth=1.5)
    ax.set_xlabel("Context Position", fontsize=12)
    ax.set_ylabel("Cumulative Variance (Top 3 PCs)", fontsize=12)
    ax.set_title("PCA Concentration vs Context Position", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pca_variance_vs_position.png", dpi=150)
    plt.close()
    print("  Saved: pca_variance_vs_position.png")

    # ---- Plot 3: Belief State R² vs Position (Mess3 only) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in ["mess3", "mess3_weak"]:
        if name not in results:
            continue
        res = results[name]
        if res["belief_r2_by_pos"]:
            pos = sorted(res["belief_r2_by_pos"].keys())
            r2 = [res["belief_r2_by_pos"][p] for p in pos]
            axes[0].plot(pos, r2, '-o', label=name, color=colors.get(name, "gray"),
                         markersize=3, linewidth=1.5)
        if res["dist_corr_by_pos"]:
            pos = sorted(res["dist_corr_by_pos"].keys())
            dc = [res["dist_corr_by_pos"][p] for p in pos]
            axes[1].plot(pos, dc, '-o', label=name, color=colors.get(name, "gray"),
                         markersize=3, linewidth=1.5)

    axes[0].set_xlabel("Context Position")
    axes[0].set_ylabel("R² (Belief State Regression)")
    axes[0].set_title("Belief State Recovery vs Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Context Position")
    axes[1].set_ylabel("R² (Pairwise Distance Correlation)")
    axes[1].set_title("Geometric Isometry vs Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "belief_state_regression.png", dpi=150)
    plt.close()
    print("  Saved: belief_state_regression.png")

    # ---- Plot 4: Probe Accuracy vs Position ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        if not res["probe_by_pos"]:
            continue
        pos = sorted(res["probe_by_pos"].keys())
        accs = [res["probe_by_pos"][p][0] for p in pos]
        stds = [res["probe_by_pos"][p][1] for p in pos]
        ax.errorbar(pos, accs, yerr=stds, fmt='-o', label=name,
                     color=colors.get(name, "gray"), markersize=3, linewidth=1.5,
                     capsize=2)
    ax.set_xlabel("Context Position", fontsize=12)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=12)
    ax.set_title("State Prediction Accuracy vs Context Position", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "probe_accuracy_vs_position.png", dpi=150)
    plt.close()
    print("  Saved: probe_accuracy_vs_position.png")

    # ---- Plot 5: Layer-wise Participation Ratio ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        layers = sorted(res["layer_pr"].keys())
        prs = [res["layer_pr"][l] for l in layers]
        ax.plot(layers, prs, '-o', label=name, color=colors.get(name, "gray"),
                markersize=5, linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Average Participation Ratio", fontsize=12)
    ax.set_title("Effective Dimensionality Across Layers", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "layer_participation_ratio.png", dpi=150)
    plt.close()
    print("  Saved: layer_participation_ratio.png")

    # ---- Plot 6: Training curves ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (name, hist) in enumerate(histories.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.plot(hist["train_loss"], label="train", alpha=0.8)
        ax.plot(hist["val_loss"], label="val", alpha=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print("  Saved: training_curves.png")

    # ---- Plot 7: Normalized PR (relative change from position 0) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        if name == "tictactoe":
            continue
        positions = res["positions"]
        pr = res["participation_ratio"]
        pr_normalized = [p / pr[0] if pr[0] > 0 else 1.0 for p in pr]
        ax.plot(positions, pr_normalized, '-o', label=name, color=colors.get(name, "gray"),
                markersize=3, linewidth=1.5)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='baseline (pos 0)')
    ax.set_xlabel("Context Position", fontsize=12)
    ax.set_ylabel("Normalized PR (relative to position 0)", fontsize=12)
    ax.set_title("Relative Dimensionality Change Across Context", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "normalized_pr_vs_position.png", dpi=150)
    plt.close()
    print("  Saved: normalized_pr_vs_position.png")


# ============================================================================
# Statistical Analysis
# ============================================================================
def statistical_analysis(results):
    """Compute statistical tests for the hypotheses."""
    from scipy import stats

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    stats_results = {}

    for name, res in results.items():
        print(f"\n--- {name} ---")
        positions = np.array(res["positions"])
        pr = np.array(res["participation_ratio"])

        # H1: PR decreases with position (Spearman correlation)
        if len(positions) > 5:
            rho, p_val = stats.spearmanr(positions, pr)
            print(f"  H1 (PR vs position): rho={rho:.3f}, p={p_val:.4f}")

            # Split into early (first quarter) vs late (last quarter)
            n = len(positions)
            q = n // 4
            early_pr = pr[:q]
            late_pr = pr[-q:]
            if len(early_pr) > 1 and len(late_pr) > 1:
                t_stat, t_pval = stats.ttest_ind(early_pr, late_pr)
                cohens_d = (early_pr.mean() - late_pr.mean()) / np.sqrt(
                    (early_pr.std()**2 + late_pr.std()**2) / 2
                )
                print(f"  Early vs Late PR: t={t_stat:.3f}, p={t_pval:.4f}, Cohen's d={cohens_d:.3f}")
                print(f"    Early mean: {early_pr.mean():.3f}, Late mean: {late_pr.mean():.3f}")

            stats_results[name] = {
                "pr_spearman_rho": float(rho),
                "pr_spearman_p": float(p_val),
                "pr_early_mean": float(early_pr.mean()) if len(early_pr) > 0 else None,
                "pr_late_mean": float(late_pr.mean()) if len(late_pr) > 0 else None,
            }

        # Belief state R² trend
        if res["belief_r2_by_pos"]:
            pos_b = sorted(res["belief_r2_by_pos"].keys())
            r2_vals = [res["belief_r2_by_pos"][p] for p in pos_b]
            rho_b, p_b = stats.spearmanr(pos_b, r2_vals)
            print(f"  Belief R² vs position: rho={rho_b:.3f}, p={p_b:.4f}")
            stats_results[name]["belief_r2_spearman_rho"] = float(rho_b)
            stats_results[name]["belief_r2_spearman_p"] = float(p_b)

        # Probe accuracy trend
        if res["probe_by_pos"]:
            pos_p = sorted(res["probe_by_pos"].keys())
            acc_vals = [res["probe_by_pos"][p][0] for p in pos_p]
            if len(pos_p) > 3:
                rho_p, p_p = stats.spearmanr(pos_p, acc_vals)
                print(f"  Probe acc vs position: rho={rho_p:.3f}, p={p_p:.4f}")
                stats_results[name]["probe_spearman_rho"] = float(rho_p)
                stats_results[name]["probe_spearman_p"] = float(p_p)

    return stats_results


# ============================================================================
# Random Baseline
# ============================================================================
def random_baseline_analysis(datasets):
    """Analyze geometry from an untrained (random) model as baseline."""
    print("\n" + "=" * 60)
    print("RANDOM MODEL BASELINE")
    print("=" * 60)

    baseline_results = {}

    # Use mess3 and rps_counter as representative
    for name in ["mess3", "rps_counter"]:
        ds = datasets[name]
        config = MODEL_CONFIG.copy()
        model = SmallGPT(ds["vocab_size"], config["d_model"], config["n_heads"],
                         config["n_layers"], ds["emissions"].shape[1]).to(DEVICE)
        model.eval()

        data = ds["emissions"][:500, :-1]
        X = torch.tensor(data, dtype=torch.long).to(DEVICE)
        streams = model.get_all_residual_streams(X)

        final_layer = config["n_layers"]
        geom = analyze_geometry_by_position(streams, final_layer)
        positions = sorted(geom.keys())
        pr = [geom[p]["participation_ratio"] for p in positions]

        baseline_results[f"{name}_random"] = {
            "positions": positions,
            "participation_ratio": pr,
        }
        print(f"  {name} random model PR: {pr[0]:.2f} -> {pr[-1]:.2f}")

    return baseline_results


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()

    print("=" * 60)
    print("MANY-SHOT RESIDUAL STREAM GEOMETRY EXPERIMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Phase 1: Generate data
    datasets = generate_all_data()

    # Phase 2: Train models
    models, histories = train_all_models(datasets)

    # Phase 3: Extract activations
    all_streams = extract_activations(models, datasets)

    # Phase 4: Analyze geometry
    results = analyze_all(all_streams, datasets, models)

    # Phase 5: Random baseline
    baseline_results = random_baseline_analysis(datasets)

    # Phase 6: Statistical analysis
    stats_results = statistical_analysis(results)

    # Phase 7: Visualize
    plot_results(results, histories)

    # Save all results
    save_data = {
        "results": {},
        "baseline": {},
        "stats": stats_results,
        "config": MODEL_CONFIG,
    }
    for name, res in results.items():
        save_data["results"][name] = {
            "positions": res["positions"],
            "participation_ratio": res["participation_ratio"],
            "pca_var_3": res["pca_var_3"],
            "pca_var_5": res["pca_var_5"],
            "layer_pr": {str(k): v for k, v in res["layer_pr"].items()},
            "belief_r2_by_pos": {str(k): v for k, v in res["belief_r2_by_pos"].items()},
            "dist_corr_by_pos": {str(k): v for k, v in res["dist_corr_by_pos"].items()},
            "probe_by_pos": {str(k): list(v) for k, v in res["probe_by_pos"].items()},
        }
    for name, res in baseline_results.items():
        save_data["baseline"][name] = {
            "positions": res["positions"],
            "participation_ratio": res["participation_ratio"],
        }

    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: results/experiment_results.json")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
