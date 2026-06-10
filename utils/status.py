"""Human-readable project and experiment status."""

import os

import torch

from utils.checkpoints import checkpoint_paths, load_latest_checkpoint, make_agent
from utils.logger import ExperimentLogger


def experiment_status(checkpoint_dir="checkpoints"):
    """Return status information for one experiment directory."""
    checkpoint, checkpoint_path = load_latest_checkpoint(checkpoint_dir)
    config = checkpoint["config"]
    agent = make_agent(checkpoint)
    update = int(checkpoint.get("update", config.get("updates", 0)))
    target = int(config.get("updates", update))
    log_records = ExperimentLogger(checkpoint_paths(checkpoint_dir)["log"]).load()
    return {
        "checkpoint": checkpoint_path,
        "update": update,
        "target_updates": target,
        "progress": update / max(target, 1),
        "seed": config.get("seed"),
        "architecture": config.get("architecture", "unknown"),
        "parameters_per_agent": sum(parameter.numel() for parameter in agent.parameters()),
        "training_log_records": len(log_records),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_checkpoint_exists": os.path.exists(checkpoint_paths(checkpoint_dir)["final"]),
    }


def print_status(checkpoint_dir="checkpoints"):
    """Print a concise experiment status report."""
    status = experiment_status(checkpoint_dir)
    print("\n=== EXPERIMENT STATUS ===")
    print(f"Checkpoint       : {status['checkpoint']}")
    print(
        f"Training progress: {status['update']}/{status['target_updates']} "
        f"({status['progress']:.1%})"
    )
    seed = status["seed"] if status["seed"] is not None else "legacy/unknown"
    print(f"Seed             : {seed}")
    print(f"Architecture     : {status['architecture']}")
    print(f"Parameters/agent : {status['parameters_per_agent']:,}")
    print(f"JSON log records : {status['training_log_records']}")
    print(f"CUDA available   : {status['cuda_available']}")
    if status["gpu"]:
        print(f"GPU              : {status['gpu']}")
    return status
