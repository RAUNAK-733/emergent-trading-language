"""Measure how actively each message position uses the available vocabulary."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def positional_entropy(messages, vocab_size, msg_length):
    """Return Shannon entropy and usage status for each message position."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2.")
    if msg_length < 1:
        raise ValueError("msg_length must be at least 1.")

    entropies = []
    max_entropy = np.log2(vocab_size)

    for position in range(msg_length):
        symbols = [message[position] for message in messages if len(message) > position]
        if any(symbol < 0 or symbol >= vocab_size for symbol in symbols):
            raise ValueError("messages contain a symbol outside the configured vocabulary.")

        counts = np.bincount(symbols, minlength=vocab_size).astype(float)
        total = counts.sum()
        probabilities = counts / total if total else np.zeros(vocab_size)
        nonzero = probabilities > 0
        entropy = -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
        usage_ratio = float(entropy / max_entropy)

        if usage_ratio > 0.5:
            status = "ACTIVE"
        elif usage_ratio > 0.2:
            status = "WEAK"
        else:
            status = "DEAD"

        entropies.append(
            {
                "position": position,
                "entropy": float(entropy),
                "max_entropy": float(max_entropy),
                "usage_ratio": usage_ratio,
                "status": status,
            }
        )

    return entropies


def plot_entropy(entropies, save_path="figures/positional_entropy.png"):
    """Save a positional-entropy bar chart."""
    if not entropies:
        raise ValueError("entropies must contain at least one position.")

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    positions = [entry["position"] for entry in entropies]
    values = [entry["entropy"] for entry in entropies]
    max_entropy = entropies[0]["max_entropy"]
    colors = [
        "green"
        if entry["status"] == "ACTIVE"
        else "orange"
        if entry["status"] == "WEAK"
        else "red"
        for entry in entropies
    ]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(positions, values, color=colors, alpha=0.8, edgecolor="black")
    plt.axhline(
        y=max_entropy,
        color="blue",
        linestyle="--",
        label=f"Max entropy ({max_entropy:.2f})",
    )
    plt.axhline(
        y=max_entropy * 0.5,
        color="orange",
        linestyle=":",
        label="50% threshold",
    )
    for bar, entry in zip(bars, entropies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{entry['usage_ratio']:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Token Position")
    plt.ylabel("Shannon Entropy (bits)")
    plt.title("Positional Entropy - Is Each Symbol Position Being Used?")
    plt.xticks(positions)
    plt.ylim(0, max_entropy * 1.18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved -> {save_path}")


def load_agent():
    """Load Agent A from the current team-reward checkpoint."""
    from agents.agent import Agent
    from utils.checkpoints import load_latest_checkpoint

    checkpoint, _ = load_latest_checkpoint()
    config = checkpoint["config"]
    agent = Agent(
        obs_dim=2 * config["n_resources"],
        vocab_size=config["vocab_size"],
        msg_length=config["msg_length"],
        n_resources=config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )
    agent.load_state_dict(checkpoint["agent_a"])
    agent.eval()
    return agent, config


def main():
    """Sample deterministic messages and report positional entropy."""
    from env.trading_env import TradingEnv

    agent_a, config = load_agent()
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    messages = []

    with torch.no_grad():
        for _ in range(500):
            obs_a_np, _ = env.reset()
            obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
            message, _ = agent_a.speak(
                obs_a,
                temperature=0.01,
                deterministic=True,
            )
            symbols = message.squeeze(0).argmax(dim=-1).tolist()
            if isinstance(symbols, int):
                symbols = [symbols]
            messages.append(symbols)

    results = positional_entropy(
        messages,
        vocab_size=config["vocab_size"],
        msg_length=config["msg_length"],
    )
    print("\n=== POSITIONAL ENTROPY RESULTS ===")
    for entry in results:
        print(
            f"Position {entry['position']}: H={entry['entropy']:.3f} bits "
            f"({entry['usage_ratio']:.0%} of max) - {entry['status']}"
        )

    plot_entropy(results)
    print("\nIf usage > 50%, the position actively distinguishes meanings.")
    print("Active positions are evidence of symbol use, not proof of grammar by themselves.")


if __name__ == "__main__":
    main()
