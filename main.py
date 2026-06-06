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
    subparsers.add_parser("verify", help="Run communication-control verification.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "baseline":
        from env.baseline import evaluate_baseline

        evaluate_baseline(n_episodes=args.episodes)
    elif args.command == "train":
        from training.train import train

        train(resume=not args.fresh)
    elif args.command == "verify":
        from analysis.verify import verify

        verify()


if __name__ == "__main__":
    main()
