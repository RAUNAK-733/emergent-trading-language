"""UMAP visualization of the speaker's learned representation."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import TradingEnv
from utils.checkpoints import load_latest_checkpoint, make_agent


def collect_speaker_representations(checkpoint_dir="checkpoints", n_samples=1500):
    """Collect speaker hidden states, symbols, and preferred resources."""
    checkpoint, _ = load_latest_checkpoint(checkpoint_dir)
    config = checkpoint["config"]
    agent = make_agent(checkpoint)
    env = TradingEnv(config["n_resources"], config["max_inventory"])
    representations, symbols, preferences = [], [], []

    with torch.no_grad():
        for _ in range(n_samples):
            obs_a_np, _ = env.reset()
            obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
            hidden = agent.speak_net[:2](obs_a)
            message, _ = agent.speak(obs_a, deterministic=True)
            representations.append(hidden.squeeze(0).numpy())
            symbols.append(int(message.squeeze().argmax().item()))
            preferences.append(int(np.argmax(env.util_a)))
    return np.asarray(representations), np.asarray(symbols), np.asarray(preferences), config


def plot_projection(projection, symbols, preferences, save_path):
    """Plot a 2D representation colored by symbol and private preference."""
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    symbol_plot = axes[0].scatter(
        projection[:, 0],
        projection[:, 1],
        c=symbols,
        cmap="tab10",
        s=12,
        alpha=0.65,
    )
    axes[0].set_title("Speaker representation by message symbol")
    figure.colorbar(symbol_plot, ax=axes[0], label="Symbol")

    preference_plot = axes[1].scatter(
        projection[:, 0],
        projection[:, 1],
        c=preferences,
        cmap="Set1",
        s=12,
        alpha=0.65,
    )
    axes[1].set_title("Speaker representation by preferred resource")
    figure.colorbar(preference_plot, ax=axes[1], label="Preferred resource")

    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.15)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def main(checkpoint_dir="checkpoints", figure_dir="figures"):
    """Run UMAP and save the speaker-representation visualization."""
    import umap

    representations, symbols, preferences, config = collect_speaker_representations(
        checkpoint_dir
    )
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=config.get("seed", 7),
    )
    projection = reducer.fit_transform(representations)
    save_path = os.path.join(figure_dir, "speaker_umap.png")
    plot_projection(projection, symbols, preferences, save_path)
    print(f"UMAP visualization saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    main()
