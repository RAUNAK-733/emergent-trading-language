"""Measure topographic similarity between agent meanings and messages."""

import os
import sys

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


def topographic_similarity(meanings, messages):
    """Return Spearman correlation between meaning and message distances."""
    if len(meanings) != len(messages):
        raise ValueError("meanings and messages must contain the same number of items.")
    if len(meanings) < 2:
        return 0.0, 1.0

    meaning_dists = []
    message_dists = []
    n = len(meanings)

    for i in range(n):
        for j in range(i + 1, n):
            meaning_a = np.asarray(meanings[i], dtype=float)
            meaning_b = np.asarray(meanings[j], dtype=float)
            meaning_distance = float(cosine(meaning_a, meaning_b + 1e-8))

            message_distance = sum(
                token_a != token_b
                for token_a, token_b in zip(messages[i], messages[j])
            )
            message_distance += abs(len(messages[i]) - len(messages[j]))
            normalized_message_distance = message_distance / max(
                len(messages[i]),
                len(messages[j]),
                1,
            )

            meaning_dists.append(meaning_distance)
            message_dists.append(normalized_message_distance)

    if len(set(message_dists)) < 2:
        return 0.0, 1.0

    correlation, pvalue = spearmanr(meaning_dists, message_dists)
    if np.isnan(correlation) or np.isnan(pvalue):
        return 0.0, 1.0
    return float(correlation), float(pvalue)


def interpret(score):
    """Describe the strength of the measured language structure."""
    if score > 0.4:
        return "STRONG - compositional language emerged"
    if score > 0.2:
        return "MODERATE - some structure present"
    if score > 0.05:
        return "WEAK - minimal structure"
    return "NONE - random babbling"


def load_agent():
    """Load Agent A from the current team-reward checkpoint."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.agent import Agent
    from utils.checkpoints import load_latest_checkpoint

    import torch

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
    """Sample meanings and messages, then print their topographic similarity."""
    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.trading_env import TradingEnv

    agent_a, config = load_agent()
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    meanings, messages = [], []

    with torch.no_grad():
        for _ in range(300):
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
            meanings.append(obs_a_np)
            messages.append(symbols)

    score, pvalue = topographic_similarity(meanings, messages)
    print(f"Topsim score : {score:.4f}")
    print(f"P-value      : {pvalue:.4f}")
    print(f"Result       : {interpret(score)}")


if __name__ == "__main__":
    main()
