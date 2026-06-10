"""Linear probes for information encoded in learned messages."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import TradingEnv
from utils.checkpoints import load_latest_checkpoint, make_agent


def fit_probe(features, targets, seed=7):
    """Fit a linear classifier and compare it with a majority baseline."""
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be 2D and align with targets.")
    if len(np.unique(targets)) < 2:
        return {
            "accuracy": 1.0,
            "baseline_accuracy": 1.0,
            "advantage": 0.0,
            "confusion_matrix": [[len(targets)]],
        }

    stratify = targets if np.min(np.bincount(targets)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    majority = np.bincount(y_train).argmax()
    baseline_predictions = np.full_like(y_test, majority)
    accuracy = float(accuracy_score(y_test, predictions))
    baseline_accuracy = float(accuracy_score(y_test, baseline_predictions))
    labels = sorted(np.unique(targets).tolist())
    return {
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "advantage": accuracy - baseline_accuracy,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
    }


def collect_message_dataset(checkpoint_dir="checkpoints", n_samples=3000):
    """Collect one-hot messages and the sender's preferred resource."""
    checkpoint, _ = load_latest_checkpoint(checkpoint_dir)
    config = checkpoint["config"]
    agent = make_agent(checkpoint)
    env = TradingEnv(config["n_resources"], config["max_inventory"])
    features, preferred_resources = [], []

    with torch.no_grad():
        for _ in range(n_samples):
            obs_a_np, _ = env.reset()
            obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
            message, _ = agent.speak(obs_a, deterministic=True)
            features.append(message.flatten().numpy())
            preferred_resources.append(int(np.argmax(env.util_a)))
    return np.asarray(features), np.asarray(preferred_resources), config


def plot_confusion(matrix, save_path):
    """Save a confusion matrix for the utility-preference probe."""
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted preferred resource")
    plt.ylabel("True preferred resource")
    plt.title("Linear probe: utility preference from message")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(checkpoint_dir="checkpoints", figure_dir="figures"):
    """Run and report a linear probe on learned messages."""
    features, targets, config = collect_message_dataset(checkpoint_dir)
    result = fit_probe(features, targets, seed=config.get("seed", 7))
    save_path = os.path.join(figure_dir, "utility_probe_confusion.png")
    plot_confusion(result["confusion_matrix"], save_path)

    print("\n=== LINEAR PROBE RESULTS ===")
    print(f"Message probe accuracy : {result['accuracy']:.1%}")
    print(f"Majority baseline      : {result['baseline_accuracy']:.1%}")
    print(f"Probe advantage        : {result['advantage']:+.1%}")
    print(f"Saved -> {save_path}")
    return result


if __name__ == "__main__":
    main()
