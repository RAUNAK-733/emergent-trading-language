"""Shared checkpoint paths and loading helpers."""

import os

import torch


TRAINING_STATE_PATH = "checkpoints/give_based_team_reward_state.pt"
FINAL_CHECKPOINT_PATH = "checkpoints/give_based_team_reward.pt"


def latest_checkpoint_path(checkpoint_dir="checkpoints"):
    """Return the newest available team-reward checkpoint path."""
    candidates = [
        os.path.join(checkpoint_dir, os.path.basename(TRAINING_STATE_PATH)),
        os.path.join(checkpoint_dir, os.path.basename(FINAL_CHECKPOINT_PATH)),
    ]
    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("Run team-reward training before analysis.")
    return max(existing, key=os.path.getmtime)


def load_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Load the newest trusted local team-reward checkpoint."""
    path = latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, path
