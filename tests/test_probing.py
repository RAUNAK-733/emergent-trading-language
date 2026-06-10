"""Tests for message probing."""

import unittest

import numpy as np

from analysis.probing import fit_probe


class ProbingTests(unittest.TestCase):
    def test_probe_detects_encoded_classes(self):
        features = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        targets = np.array([0, 0, 0, 1, 1, 1])

        result = fit_probe(features, targets, seed=7)

        self.assertGreater(result["accuracy"], result["baseline_accuracy"])

    def test_probe_validates_shapes(self):
        with self.assertRaises(ValueError):
            fit_probe([[1, 0]], [0, 1])
