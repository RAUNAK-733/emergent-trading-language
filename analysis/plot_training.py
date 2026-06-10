"""Plot training efficiency and language advantage from JSON logs."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_training_logs(log_dir="checkpoints"):
    """Load and sort training records from checkpoint JSON logs."""
    records = []
    log_files = sorted(glob.glob(os.path.join(log_dir, "log_*.json")))

    for log_file in log_files:
        with open(log_file, encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
        else:
            raise ValueError(f"Unsupported log format in {log_file}.")

    if not records:
        log_file = os.path.join(log_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"Unsupported log format in {log_file}.")

    required = {"update", "efficiency", "lang_adv"}
    valid_records = []
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            continue
        valid_records.append(record)

    return sorted(valid_records, key=lambda record: record["update"])


def plot_training_curve(
    log_dir="checkpoints",
    save_path="figures/training_curve.png",
):
    """Plot normal efficiency, no-message efficiency, and language advantage."""
    records = load_training_logs(log_dir)
    if not records:
        print("No log data found. Skipping plot.")
        return None

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    updates = [record["update"] for record in records]
    efficiencies = [record["efficiency"] for record in records]
    language_advantages = [record["lang_adv"] for record in records]
    no_message_efficiencies = [record.get("no_msg", 0.0) for record in records]
    random_baseline = records[-1].get("random_baseline", 0.007)

    figure, (efficiency_axis, language_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
    )

    efficiency_axis.plot(
        updates,
        efficiencies,
        "b-",
        label="Normal messages",
        linewidth=1.5,
    )
    efficiency_axis.plot(
        updates,
        no_message_efficiencies,
        "r--",
        label="No messages",
        linewidth=1.5,
    )
    efficiency_axis.axhline(
        y=random_baseline,
        color="gray",
        linestyle=":",
        label=f"Random baseline ({random_baseline:.3f})",
    )
    efficiency_axis.set_ylabel("Efficiency")
    efficiency_axis.legend()
    efficiency_axis.set_title("Training Progress - Emergent Language Negotiator")
    efficiency_axis.grid(alpha=0.2)

    language_axis.plot(
        updates,
        language_advantages,
        "g-",
        label="Language advantage",
        linewidth=1.5,
    )
    language_axis.axhline(
        y=0.02,
        color="orange",
        linestyle="--",
        label="Minimum threshold (0.02)",
    )
    language_axis.set_ylabel("Language Advantage")
    language_axis.set_xlabel("Training Update")
    language_axis.legend()
    language_axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    print(f"Training curve saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    plot_training_curve()
