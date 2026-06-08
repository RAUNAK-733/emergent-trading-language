"""Tests for positional-entropy analysis."""

import os
import tempfile
import unittest

from analysis.entropy import plot_entropy, positional_entropy


class PositionalEntropyTests(unittest.TestCase):
    def test_uniform_symbols_reach_maximum_entropy(self):
        results = positional_entropy([[0], [1], [2], [3]], vocab_size=4, msg_length=1)

        self.assertAlmostEqual(results[0]["entropy"], 2.0)
        self.assertAlmostEqual(results[0]["usage_ratio"], 1.0)
        self.assertEqual(results[0]["status"], "ACTIVE")

    def test_constant_position_is_dead(self):
        results = positional_entropy([[1], [1], [1]], vocab_size=4, msg_length=1)

        self.assertEqual(results[0]["entropy"], 0.0)
        self.assertEqual(results[0]["status"], "DEAD")

    def test_missing_positions_are_handled(self):
        results = positional_entropy([[0], [1, 2], []], vocab_size=3, msg_length=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["status"], "DEAD")

    def test_invalid_symbol_raises(self):
        with self.assertRaises(ValueError):
            positional_entropy([[4]], vocab_size=4, msg_length=1)

    def test_plot_is_saved(self):
        results = positional_entropy([[0], [1]], vocab_size=2, msg_length=1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "entropy.png")
            plot_entropy(results, path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
