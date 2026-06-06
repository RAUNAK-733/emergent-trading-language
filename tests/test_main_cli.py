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
        )

    def test_default_train_uses_training_default(self):
        with patch("sys.argv", ["main.py", "train"]):
            with patch("training.train.train") as train:
                main.main()

        train.assert_called_once_with(resume=True, n_updates=25000)


if __name__ == "__main__":
    unittest.main()
