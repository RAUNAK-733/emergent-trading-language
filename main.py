"""Command-line entry point for the emergent trading language project."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate emergent communication trading agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the blind random baseline.")
    baseline.add_argument("--episodes", type=int, default=2000)

    train_parser = subparsers.add_parser("train", help="Train or resume agents.")
    train_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved training progress and start a new run.",
    )
    train_parser.add_argument(
        "--updates",
        type=int,
        default=25000,
        help="Override the default number of training updates.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for a fresh training run.",
    )
    subparsers.add_parser("verify", help="Run communication-control verification.")
    subparsers.add_parser("analyze", help="Run all language analyses and plots.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "baseline":
        from env.baseline import evaluate_baseline

        evaluate_baseline(n_episodes=args.episodes)
    elif args.command == "train":
        from training.train import train

        train(resume=not args.fresh, n_updates=args.updates, seed=args.seed)
    elif args.command == "verify":
        from analysis.verify import verify

        verify()
    elif args.command == "analyze":
        from analysis.entropy import main as entropy
        from analysis.plot_training import plot_training_curve
        from analysis.topsim import main as topsim
        from analysis.verify import verify

        verify()
        topsim()
        entropy()
        plot_training_curve()


if __name__ == "__main__":
    main()
