"""Tests for multi-run summaries."""

import unittest

from analysis.compare_runs import summarize_reports


class CompareRunsTests(unittest.TestCase):
    def test_summary_reports_mean_and_standard_deviation(self):
        reports = [
            {
                "results": {
                    "normal": {"efficiency": 0.2},
                    "zero": {"efficiency": 0.0},
                    "random": {"efficiency": 0.05},
                },
                "minimum_language_advantage": 0.15,
            },
            {
                "results": {
                    "normal": {"efficiency": 0.4},
                    "zero": {"efficiency": 0.1},
                    "random": {"efficiency": 0.2},
                },
                "minimum_language_advantage": 0.2,
            },
        ]

        summary = summarize_reports(reports)

        self.assertAlmostEqual(summary["normal_efficiency"]["mean"], 0.3)
        self.assertGreater(summary["normal_efficiency"]["std"], 0.0)

    def test_empty_reports_raise(self):
        with self.assertRaises(ValueError):
            summarize_reports([])
