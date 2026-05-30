"""Verify trained agents with communication-control tests."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent import Agent
from env.trading_env import TradingEnv


def optimal_joint_reward(env):
    reward_a = float(np.dot(env.inv_b, env.util_a))
    reward_b = float(np.dot(env.inv_a, env.util_b))
    return max(reward_a + reward_b, 1e-8)


def score_trade(env, offer_a, offer_b):
    reward_a, reward_b, success = env.step(offer_a, offer_b)
    if not success:
        return 0, 0.0, 0
    efficiency = (reward_a + reward_b) / optimal_joint_reward(env)
    useful = int(efficiency >= 0.60)
    return 1, float(efficiency), useful


def load_config():
    default = {
        "n_resources": 2,
        "max_inventory": 5,
        "max_offer": 5,
        "vocab_size": 4,
        "msg_length": 1,
        "hidden_dim": 96,
    }
    path = "checkpoints/config.pt"
    if os.path.exists(path):
        default.update(torch.load(path, map_location="cpu"))
    return default


def make_agents(config):
    obs_dim = 2 * config["n_resources"]
    agent_a = Agent(
        obs_dim,
        vocab_size=config["vocab_size"],
        msg_length=config["msg_length"],
        n_resources=config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )
    agent_b = Agent(
        obs_dim,
        vocab_size=config["vocab_size"],
        msg_length=config["msg_length"],
        n_resources=config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )
    agent_a.load_state_dict(torch.load("checkpoints/agent_a.pt", map_location="cpu"))
    agent_b.load_state_dict(torch.load("checkpoints/agent_b.pt", map_location="cpu"))
    agent_a.eval()
    agent_b.eval()
    return agent_a, agent_b


def message_control(msg, config, mode):
    if mode == "normal":
        return msg
    if mode == "zero":
        return torch.zeros_like(msg)
    if mode == "random":
        idx = torch.randint(config["vocab_size"], (msg.shape[0], config["msg_length"]))
        return F.one_hot(idx, num_classes=config["vocab_size"]).float()
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_mode(agent_a, agent_b, config, mode, n_episodes=3000):
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    valid_log, efficiency_log, useful_log = [], [], []
    symbols_sent, utility_vals, inventory_vals = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs_a_np, obs_b_np = env.reset()
            obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
            obs_b = torch.tensor(obs_b_np, dtype=torch.float32).unsqueeze(0)

            msg_a, _ = agent_a.speak(obs_a, temperature=0.1, deterministic=True)
            msg_b, _ = agent_b.speak(obs_b, temperature=0.1, deterministic=True)
            msg_a_used = message_control(msg_a, config, mode)
            msg_b_used = message_control(msg_b, config, mode)

            action_a, _ = agent_a.act(
                obs_a,
                msg_b_used,
                temperature=0.1,
                deterministic=True,
            )
            action_b, _ = agent_b.act(
                obs_b,
                msg_a_used,
                temperature=0.1,
                deterministic=True,
            )

            offer_a = action_a.squeeze(0).numpy().astype(int)
            offer_b = action_b.squeeze(0).numpy().astype(int)
            valid, efficiency, useful = score_trade(env, offer_a, offer_b)
            valid_log.append(valid)
            efficiency_log.append(efficiency)
            useful_log.append(useful)

            if mode == "normal":
                symbols_sent.append(int(msg_a.squeeze().argmax().item()))
                utility_vals.append(float(obs_a_np[config["n_resources"]]))
                inventory_vals.append(float(obs_a_np[0]))

    return {
        "valid": float(np.mean(valid_log)),
        "efficiency": float(np.mean(efficiency_log)),
        "useful": float(np.mean(useful_log)),
        "symbols": symbols_sent,
        "utilities": utility_vals,
        "inventories": inventory_vals,
    }


def plot_symbol_heatmap(symbols, values, title, xlabel, filename, vocab_size):
    n_buckets = 5
    heatmap_data = np.zeros((vocab_size, n_buckets))
    for sym, value in zip(symbols, values):
        clipped = min(max(float(value), 0.0), 0.999999)
        bucket = min(int(clipped * n_buckets), n_buckets - 1)
        heatmap_data[sym, bucket] += 1

    col_sums = heatmap_data.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    heatmap_data = heatmap_data / col_sums

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        yticklabels=[f"Symbol {i}" for i in range(vocab_size)],
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Symbol sent")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_control_bars(results):
    labels = list(results.keys())
    efficiencies = [results[label]["efficiency"] for label in labels]
    useful = [results[label]["useful"] for label in labels]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, efficiencies, width, label="Efficiency")
    plt.bar(x + width / 2, useful, width, label="Useful trade rate")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Communication controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/communication_controls.png", dpi=150)
    plt.close()


def verify():
    os.makedirs("figures", exist_ok=True)
    config = load_config()
    agent_a, agent_b = make_agents(config)

    results = {
        "normal": evaluate_mode(agent_a, agent_b, config, "normal"),
        "zero": evaluate_mode(agent_a, agent_b, config, "zero"),
        "random": evaluate_mode(agent_a, agent_b, config, "random"),
    }

    best_control = max(results["zero"]["efficiency"], results["random"]["efficiency"])
    language_gain = results["normal"]["efficiency"] - best_control
    strong = language_gain >= 0.05 and results["normal"]["useful"] > results["zero"]["useful"]

    print("\n=== VERIFICATION RESULTS ===")
    for name, result in results.items():
        print(
            f"{name:>7} | valid={result['valid']:.1%} | "
            f"useful={result['useful']:.1%} | efficiency={result['efficiency']:.3f}"
        )
    print(f"\nLanguage advantage over controls: {language_gain:.3f}")
    print(f"Conclusion: {'COMMUNICATION HELPS' if strong else 'NOT PROVEN YET'}")

    normal = results["normal"]
    plot_symbol_heatmap(
        normal["symbols"],
        normal["utilities"],
        "Agent A symbol vs utility for resource 0",
        "Agent A utility for resource 0",
        "figures/symbol_utility_heatmap.png",
        config["vocab_size"],
    )
    scaled_inventory = [
        (value - 1) / max(config["max_inventory"] - 1, 1)
        for value in normal["inventories"]
    ]
    plot_symbol_heatmap(
        normal["symbols"],
        scaled_inventory,
        "Agent A symbol vs inventory for resource 0",
        "Agent A inventory for resource 0",
        "figures/symbol_inventory_heatmap.png",
        config["vocab_size"],
    )
    plot_control_bars(results)

    print("\nSaved figures:")
    print("figures/symbol_utility_heatmap.png")
    print("figures/symbol_inventory_heatmap.png")
    print("figures/communication_controls.png")


if __name__ == "__main__":
    verify()
