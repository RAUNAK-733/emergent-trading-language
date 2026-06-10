"""Tests for command-line training controls."""

import unittest
from unittest.mock import patch

import main


class MainCliTests(unittest.TestCase):
    def test_train_updates_override_is_forwarded(self):
        with patch("sys.argv", ["main.py", "train", "--fresh", "--updates", "50000"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(
            resume=False,
            n_updates=50000,
            seed=7,
        )

    def test_default_train_uses_training_default(self):
        with patch("sys.argv", ["main.py", "train"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(resume=True, n_updates=25000, seed=7)

    def test_seed_override_is_forwarded(self):
        with patch("sys.argv", ["main.py", "train", "--fresh", "--seed", "42"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(resume=False, n_updates=25000, seed=42)

    def test_analyze_runs_all_analysis_steps(self):
        with patch("sys.argv", ["main.py", "analyze"]):
            with patch("analysis.verify.verify") as verify:
                with patch("analysis.topsim.main") as topsim:
                    with patch("analysis.entropy.main") as entropy:
                        with patch(
                            "analysis.plot_training.plot_training_curve"
                        ) as plot_training:
                            main.main()

        verify.assert_called_once_with()
        topsim.assert_called_once_with()
        entropy.assert_called_once_with()
        plot_training.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
