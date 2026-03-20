"""
Train small GPT models on game/HMM sequences for residual stream geometry analysis.
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import SmallGPT


def train_model(name, train_data, vocab_size, config, device="cuda:0", save_dir="results/models"):
    """Train a SmallGPT on next-token prediction.

    Args:
        name: Model name for saving
        train_data: (N, seq_len) int array of token sequences
        vocab_size: Number of tokens
        config: Dict with d_model, n_heads, n_layers, epochs, lr, batch_size
        device: CUDA device
        save_dir: Where to save model checkpoints

    Returns:
        Trained model, training history
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    d_model = config.get("d_model", 64)
    n_heads = config.get("n_heads", 4)
    n_layers = config.get("n_layers", 4)
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 3e-4)
    batch_size = config.get("batch_size", 128)
    max_seq_len = train_data.shape[1]

    model = SmallGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Input = tokens[:-1], target = tokens[1:]
    X = torch.tensor(train_data[:, :-1], dtype=torch.long)
    Y = torch.tensor(train_data[:, 1:], dtype=torch.long)

    # Split 90/10
    n = len(X)
    n_train = int(0.9 * n)
    train_ds = TensorDataset(X[:n_train], Y[:n_train])
    val_ds = TensorDataset(X[n_train:], Y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Model: {param_count:,} params, d={d_model}, h={n_heads}, L={n_layers}, seq_len={max_seq_len}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        train_loss = total_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path / f"{name}_best.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Load best
    model.load_state_dict(torch.load(save_path / f"{name}_best.pt", weights_only=True))
    model.eval()

    # Save config
    with open(save_path / f"{name}_config.json", "w") as f:
        json.dump({**config, "vocab_size": vocab_size, "max_seq_len": max_seq_len,
                   "param_count": param_count, "best_val_loss": best_val_loss}, f, indent=2)

    print(f"[{name}] Done. Best val loss: {best_val_loss:.4f}")
    return model, history
