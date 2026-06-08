"""Tests for training-curve plotting."""

import json
import os
import tempfile
import unittest

from analysis.plot_training import load_training_logs, plot_training_curve


class PlotTrainingTests(unittest.TestCase):
    def test_loads_and_sorts_checkpoint_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            for update in [1000, 500]:
                path = os.path.join(directory, f"log_{update}.json")
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "update": update,
                            "efficiency": 0.1,
                            "lang_adv": 0.08,
                            "no_msg": 0.02,
                        },
                        file,
                    )

            records = load_training_logs(directory)

        self.assertEqual([record["update"] for record in records], [500, 1000])

    def test_plots_combined_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "training_log.json")
            plot_path = os.path.join(directory, "training_curve.png")
            with open(log_path, "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {
                            "update": 500,
                            "efficiency": 0.18,
                            "lang_adv": 0.17,
                            "no_msg": 0.01,
                        }
                    ],
                    file,
                )

            result = plot_training_curve(directory, plot_path)

            self.assertEqual(result, plot_path)
            self.assertTrue(os.path.exists(plot_path))

    def test_missing_logs_skip_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(plot_training_curve(directory))


if __name__ == "__main__":
    unittest.main()
