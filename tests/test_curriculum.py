"""Tests for curriculum scheduling."""

import unittest

from training.curriculum import CurriculumScheduler


class CurriculumSchedulerTests(unittest.TestCase):
    def test_offer_limit_grows_with_progress(self):
        scheduler = CurriculumScheduler(max_offer=5)

        self.assertEqual(scheduler.offer_limit(0.0), 1)
        self.assertEqual(scheduler.offer_limit(1.0), 5)

    def test_strict_weight_reaches_one(self):
        scheduler = CurriculumScheduler(max_offer=5, strict_fraction=0.6)

        self.assertAlmostEqual(scheduler.strict_weight(0.3), 0.5)
        self.assertEqual(scheduler.strict_weight(0.6), 1.0)

    def test_auxiliary_guidance_fades_out(self):
        scheduler = CurriculumScheduler(max_offer=5)

        self.assertEqual(scheduler.auxiliary_weight(1.0, 0.0), 1.0)
        self.assertEqual(scheduler.auxiliary_weight(1.0, 1.0), 0.0)

    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            CurriculumScheduler(max_offer=0)
