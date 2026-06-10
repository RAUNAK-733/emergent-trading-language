"""Tests for UMAP plotting helpers."""

import os
import tempfile
import unittest

import numpy as np

from analysis.umap_viz import plot_projection


class UmapVisualizationTests(unittest.TestCase):
    def test_projection_plot_is_saved(self):
        projection = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]])
        symbols = np.array([0, 1, 0])
        preferences = np.array([1, 0, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "umap.png")
            plot_projection(projection, symbols, preferences, path)

            self.assertTrue(os.path.exists(path))
