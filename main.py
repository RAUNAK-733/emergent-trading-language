"""Command-line entry point for the emergent trading language project."""

import argparse
import os


def positive_int(value):
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value):
    """Parse a nonnegative command-line integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate emergent communication trading agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the blind random baseline.")
    baseline.add_argument("--episodes", type=positive_int, default=2000)

    train_parser = subparsers.add_parser("train", help="Train or resume agents.")
    train_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved training progress and start a new run.",
    )
    train_parser.add_argument(
        "--updates",
        type=positive_int,
        default=None,
        help="Override the saved target or fresh-run default of 25,000 updates.",
    )
    train_parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=7,
        help="Random seed used for a fresh training run.",
    )
    train_parser.add_argument(
        "--run-dir",
        default="checkpoints",
        help="Directory used for this run's checkpoints and metrics.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Run communication-control verification.",
    )
    verify_parser.add_argument("--run-dir", default="checkpoints")
    verify_parser.add_argument("--figure-dir", default="figures")
    verify_parser.add_argument("--episodes", type=positive_int, default=3000)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run all language analyses and plots.",
    )
    analyze_parser.add_argument("--run-dir", default="checkpoints")
    analyze_parser.add_argument("--figure-dir", default="figures")
    analyze_parser.add_argument("--episodes", type=positive_int, default=3000)

    status_parser = subparsers.add_parser(
        "status",
        help="Show checkpoint and training status.",
    )
    status_parser.add_argument("--run-dir", default="checkpoints")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare strict verification across several run directories.",
    )
    compare_parser.add_argument("--runs", nargs="+", required=True)
    compare_parser.add_argument("--output-dir", default="figures/comparison")
    compare_parser.add_argument("--episodes", type=positive_int, default=1000)
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "baseline":
        from env.baseline import evaluate_baseline

        evaluate_baseline(n_episodes=args.episodes)
    elif args.command == "train":
        from training.train import train

        train(
            resume=not args.fresh,
            n_updates=args.updates,
            seed=args.seed,
            checkpoint_dir=args.run_dir,
        )
    elif args.command == "verify":
        from analysis.verify import verify

        verify(args.run_dir, args.figure_dir, args.episodes)
    elif args.command == "analyze":
        from analysis.entropy import main as entropy
        from analysis.plot_training import plot_training_curve
        from analysis.probing import main as probing
        from analysis.topsim import main as topsim
        from analysis.umap_viz import main as umap_viz
        from analysis.verify import verify

        verify(args.run_dir, args.figure_dir, args.episodes)
        topsim(args.run_dir)
        entropy(args.run_dir, args.figure_dir)
        probing(args.run_dir, args.figure_dir)
        umap_viz(args.run_dir, args.figure_dir)
        plot_training_curve(
            args.run_dir,
            os.path.join(args.figure_dir, "training_curve.png"),
        )
    elif args.command == "status":
        from utils.status import print_status

        print_status(args.run_dir)
    elif args.command == "compare":
        from analysis.compare_runs import compare_runs

        compare_runs(args.runs, args.output_dir, args.episodes)


if __name__ == "__main__":
    main()
