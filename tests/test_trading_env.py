"""Core behavior tests for the trading environment."""

import unittest

import numpy as np

from env.trading_env import TradingEnv


class TradingEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnv(n_resources=2, max_inventory=5)
        self.env.inv_a = np.array([5, 1])
        self.env.inv_b = np.array([1, 5])
        self.env.util_a = np.array([0.1, 0.9])
        self.env.util_b = np.array([0.9, 0.1])

    def test_reset_returns_expected_shapes(self):
        obs_a, obs_b = self.env.reset()

        self.assertEqual(obs_a.shape, (4,))
        self.assertEqual(obs_b.shape, (4,))
        self.assertAlmostEqual(float(self.env.util_a.sum()), 1.0)
        self.assertAlmostEqual(float(self.env.util_b.sum()), 1.0)

    def test_mutually_beneficial_trade_succeeds(self):
        reward_a, reward_b, success = self.env.step(
            np.array([0, 2]),
            np.array([2, 0]),
        )

        self.assertTrue(success)
        self.assertGreater(reward_a, 0.0)
        self.assertGreater(reward_b, 0.0)

    def test_harmful_trade_fails(self):
        reward_a, reward_b, success = self.env.step(
            np.array([1, 0]),
            np.array([0, 1]),
        )

        self.assertFalse(success)
        self.assertEqual(reward_a, 0.0)
        self.assertEqual(reward_b, 0.0)

    def test_unaffordable_trade_fails(self):
        reward_a, reward_b, success = self.env.step(
            np.array([2, 5]),
            np.array([2, 0]),
        )

        self.assertFalse(success)
        self.assertEqual(reward_a, 0.0)
        self.assertEqual(reward_b, 0.0)


if __name__ == "__main__":
    unittest.main()

