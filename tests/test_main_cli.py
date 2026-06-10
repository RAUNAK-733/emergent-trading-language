"""Tests for command-line training controls."""

import os
import unittest
from unittest.mock import patch

import main


class MainCliTests(unittest.TestCase):
    def test_positive_integer_validation(self):
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["main.py", "baseline", "--episodes", "0"]):
                main.main()

    def test_train_updates_override_is_forwarded(self):
        with patch("sys.argv", ["main.py", "train", "--fresh", "--updates", "50000"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(
            resume=False,
            n_updates=50000,
            seed=7,
            checkpoint_dir="checkpoints",
        )

    def test_default_train_uses_training_default(self):
        with patch("sys.argv", ["main.py", "train"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(
            resume=True,
            n_updates=None,
            seed=7,
            checkpoint_dir="checkpoints",
        )

    def test_seed_override_is_forwarded(self):
        with patch("sys.argv", ["main.py", "train", "--fresh", "--seed", "42"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(
            resume=False,
            n_updates=None,
            seed=42,
            checkpoint_dir="checkpoints",
        )

    def test_analyze_runs_all_analysis_steps(self):
        with patch("sys.argv", ["main.py", "analyze"]):
            with patch("analysis.verify.verify") as verify:
                with patch("analysis.topsim.main") as topsim:
                    with patch("analysis.entropy.main") as entropy:
                        with patch("analysis.probing.main") as probing:
                            with patch("analysis.umap_viz.main") as umap_viz:
                                with patch(
                                    "analysis.plot_training.plot_training_curve"
                                ) as plot_training:
                                    main.main()

        verify.assert_called_once_with("checkpoints", "figures", 3000)
        topsim.assert_called_once_with("checkpoints")
        entropy.assert_called_once_with("checkpoints", "figures")
        probing.assert_called_once_with("checkpoints", "figures")
        umap_viz.assert_called_once_with("checkpoints", "figures")
        plot_training.assert_called_once_with(
            "checkpoints",
            os.path.join("figures", "training_curve.png"),
        )


if __name__ == "__main__":
    unittest.main()
