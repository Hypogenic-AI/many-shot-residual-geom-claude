"""
Additional analysis: PCA scatter plots and heatmaps for residual stream geometry.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "hmm"))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "games"))

from model import SmallGPT
from generate_hmm_data import Mess3, GenericHMM, generate_dataset
from generate_game_data import RPSGenerator

DEVICE = "cuda:0"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"
MODELS_DIR = Path(__file__).parent.parent / "results" / "models"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_model(name, vocab_size, max_seq_len, d_model=64, n_heads=4, n_layers=4):
    model = SmallGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}_best.pt", weights_only=True))
    model.eval()
    return model


def pca_scatter_early_vs_late():
    """PCA scatter of residual stream at early vs late positions."""
    print("Creating PCA scatter plots...")

    # Load Mess3 model and data
    mess3 = Mess3(emission_prob=0.9)
    data = generate_dataset("mess3", 2000, 128, seed=SEED)
    model = load_model("mess3", 3, 128)

    X = torch.tensor(data["emissions"][:1000, :-1], dtype=torch.long).to(DEVICE)
    streams = model.get_all_residual_streams(X)
    final_acts = streams[4].numpy()  # Layer 4 (final block)
    states = data["states"][:1000, :-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    positions = [0, 5, 20, 50, 100, 126]
    colors_map = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}

    for idx, pos in enumerate(positions):
        ax = axes.flat[idx]
        acts = final_acts[:, pos, :]
        pca = PCA(n_components=2)
        proj = pca.fit_transform(acts)

        s = states[:, pos]
        for state_id in range(3):
            mask = s == state_id
            ax.scatter(proj[mask, 0], proj[mask, 1], c=colors_map[state_id],
                       alpha=0.3, s=5, label=f"State {state_id}")
        ax.set_title(f"Position {pos}\n(var={pca.explained_variance_ratio_[:2].sum():.2f})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if idx == 0:
            ax.legend(markerscale=3)

    fig.suptitle("Mess3: Residual Stream PCA at Different Context Positions", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pca_scatter_mess3_positions.png", dpi=150)
    plt.close()
    print("  Saved: pca_scatter_mess3_positions.png")

    # RPS counter
    rps_data = RPSGenerator.generate_dataset(2000, rounds_per_game=64, strategy="counter", seed=SEED)
    model_rps = load_model("rps_counter", 3, 128)
    X_rps = torch.tensor(rps_data[:1000, :-1], dtype=torch.long).to(DEVICE)
    streams_rps = model_rps.get_all_residual_streams(X_rps)
    final_rps = streams_rps[4].numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, pos in enumerate(positions):
        if pos >= final_rps.shape[1]:
            continue
        ax = axes.flat[idx]
        acts = final_rps[:, pos, :]
        pca = PCA(n_components=2)
        proj = pca.fit_transform(acts)

        tokens = rps_data[:1000, pos]
        token_names = ['R', 'P', 'S']
        for tok in range(3):
            mask = tokens == tok
            ax.scatter(proj[mask, 0], proj[mask, 1], alpha=0.3, s=5,
                       label=token_names[tok], c=colors_map[tok])
        ax.set_title(f"Position {pos}\n(var={pca.explained_variance_ratio_[:2].sum():.2f})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if idx == 0:
            ax.legend(markerscale=3)

    fig.suptitle("RPS Counter: Residual Stream PCA at Different Context Positions", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pca_scatter_rps_counter_positions.png", dpi=150)
    plt.close()
    print("  Saved: pca_scatter_rps_counter_positions.png")


def dimensionality_heatmap():
    """Heatmap of PR across layers and positions."""
    print("Creating dimensionality heatmaps...")

    for name, vocab, strat in [("mess3", 3, None), ("rps_counter", 3, "counter")]:
        if strat:
            data = RPSGenerator.generate_dataset(2000, rounds_per_game=64, strategy=strat, seed=SEED)
        else:
            d = generate_dataset("mess3", 2000, 128, seed=SEED)
            data = d["emissions"]

        model = load_model(name, vocab, 128)
        X = torch.tensor(data[:500, :-1], dtype=torch.long).to(DEVICE)
        streams = model.get_all_residual_streams(X)

        n_layers = 6
        positions = list(range(0, X.shape[1], max(1, X.shape[1] // 25)))
        heatmap = np.zeros((n_layers, len(positions)))

        for li in range(n_layers):
            acts = streams[li].numpy()
            for pi, pos in enumerate(positions):
                a = acts[:, pos, :]
                centered = a - a.mean(axis=0)
                cov = np.cov(centered.T)
                eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
                s = eigvals.sum()
                if s < 1e-12:
                    heatmap[li, pi] = 1.0
                else:
                    heatmap[li, pi] = s**2 / (eigvals**2).sum()

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(heatmap, aspect='auto', cmap='viridis',
                        extent=[positions[0], positions[-1], n_layers - 0.5, -0.5])
        ax.set_xlabel("Context Position")
        ax.set_ylabel("Layer")
        ax.set_title(f"{name}: Participation Ratio (Layers × Positions)")
        plt.colorbar(im, label="Participation Ratio")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"pr_heatmap_{name}.png", dpi=150)
        plt.close()
        print(f"  Saved: pr_heatmap_{name}.png")


def summary_comparison_plot():
    """Create summary comparison figure for the report."""
    print("Creating summary comparison plot...")

    with open(Path(__file__).parent.parent / "results" / "experiment_results.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: PR vs position for all datasets
    ax = axes[0]
    for name, res in data["results"].items():
        if name == "tictactoe":
            continue
        ax.plot(res["positions"], res["participation_ratio"],
                '-', linewidth=1.5, label=name, markersize=2)
    ax.set_xlabel("Context Position")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("A. Effective Dimensionality vs Context")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Belief state R² for Mess3
    ax = axes[1]
    for name in ["mess3", "mess3_weak"]:
        if name in data["results"] and data["results"][name]["belief_r2_by_pos"]:
            pos = sorted(data["results"][name]["belief_r2_by_pos"].keys(), key=int)
            r2 = [data["results"][name]["belief_r2_by_pos"][p] for p in pos]
            ax.plot([int(p) for p in pos], r2, '-o', label=name, markersize=3)
    ax.set_xlabel("Context Position")
    ax.set_ylabel("R² (Belief State Regression)")
    ax.set_title("B. Belief State Representation Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: TTT dimensionality collapse
    ax = axes[2]
    ttt = data["results"]["tictactoe"]
    ax.plot(ttt["positions"], ttt["participation_ratio"], '-o',
            color='purple', markersize=6, linewidth=2)
    ax.set_xlabel("Move Number")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("C. Tic-Tac-Toe: Dimensionality Collapse")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary_figure.png", dpi=200)
    plt.close()
    print("  Saved: summary_figure.png")


if __name__ == "__main__":
    pca_scatter_early_vs_late()
    dimensionality_heatmap()
    summary_comparison_plot()
    print("\nDone!")
