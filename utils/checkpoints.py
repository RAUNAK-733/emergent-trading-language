"""Shared checkpoint paths and loading helpers."""

import os

import torch


TRAINING_STATE_PATH = "checkpoints/give_based_team_reward_state.pt"
FINAL_CHECKPOINT_PATH = "checkpoints/give_based_team_reward.pt"


def checkpoint_paths(checkpoint_dir="checkpoints"):
    """Return the standard artifact paths for one experiment run."""
    return {
        "state": os.path.join(
            checkpoint_dir,
            os.path.basename(TRAINING_STATE_PATH),
        ),
        "final": os.path.join(
            checkpoint_dir,
            os.path.basename(FINAL_CHECKPOINT_PATH),
        ),
        "log": os.path.join(checkpoint_dir, "training_log.json"),
    }


def latest_checkpoint_path(checkpoint_dir="checkpoints"):
    """Return the newest available team-reward checkpoint path."""
    paths = checkpoint_paths(checkpoint_dir)
    candidates = [paths["state"], paths["final"]]
    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("Run team-reward training before analysis.")
    return max(existing, key=os.path.getmtime)


def load_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Load the newest trusted local team-reward checkpoint."""
    path = latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, path


def make_agent(checkpoint, agent_key="agent_a", device="cpu"):
    """Construct one Agent from a loaded team-reward checkpoint."""
    from agents.agent import Agent

    config = checkpoint["config"]
    agent = Agent(
        obs_dim=2 * config["n_resources"],
        vocab_size=config["vocab_size"],
        msg_length=config["msg_length"],
        n_resources=config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )
    agent.load_state_dict(checkpoint[agent_key])
    agent.to(device)
    agent.eval()
    return agent
