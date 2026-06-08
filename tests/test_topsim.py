"""Tests for topographic-similarity analysis."""

import unittest

import numpy as np

from analysis.topsim import interpret, topographic_similarity


class TopographicSimilarityTests(unittest.TestCase):
    def test_structured_messages_have_positive_similarity(self):
        meanings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
        ]
        messages = [[0, 0], [0, 0], [1, 1]]

        score, pvalue = topographic_similarity(meanings, messages)

        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(pvalue, 0.0)

    def test_constant_messages_return_neutral_result(self):
        score, pvalue = topographic_similarity(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            [[0], [0]],
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(pvalue, 1.0)

    def test_mismatched_inputs_raise(self):
        with self.assertRaises(ValueError):
            topographic_similarity([np.array([1.0])], [])

    def test_interpretation_thresholds(self):
        self.assertTrue(interpret(0.5).startswith("STRONG"))
        self.assertTrue(interpret(0.3).startswith("MODERATE"))
        self.assertTrue(interpret(0.1).startswith("WEAK"))
        self.assertTrue(interpret(0.0).startswith("NONE"))


if __name__ == "__main__":
    unittest.main()
