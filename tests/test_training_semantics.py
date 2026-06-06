"""Tests for the interpretation of actor actions during trading."""

import unittest

import numpy as np

from agents.agent import Agent
from env.trading_env import TradingEnv
from training.train import actions_to_offers, sample_episode


class TrainingSemanticsTests(unittest.TestCase):
    def test_actor_actions_are_given_to_the_other_agent(self):
        give_a_to_b = np.array([1, 2])
        give_b_to_a = np.array([3, 4])

        offer_a, offer_b = actions_to_offers(give_a_to_b, give_b_to_a)

        self.assertTrue(np.array_equal(offer_a, give_b_to_a))
        self.assertTrue(np.array_equal(offer_b, give_a_to_b))

    def test_both_agents_receive_the_same_team_credit(self):
        np.random.seed(4)
        env = TradingEnv(n_resources=2, max_inventory=5)
        agent_a = Agent(4, 4, 1, 2, max_offer=5)
        agent_b = Agent(4, 4, 1, 2, max_offer=5)

        episode = sample_episode(env, agent_a, agent_b, temperature=1.0, progress=0.1)

        self.assertAlmostEqual(
            episode["team_reward"],
            episode["reward_a"] + episode["reward_b"],
        )
        self.assertIn("logp_agent_a", episode)
        self.assertIn("logp_agent_b", episode)
        self.assertGreaterEqual(episode["average_give"], 0.0)


if __name__ == "__main__":
    unittest.main()
