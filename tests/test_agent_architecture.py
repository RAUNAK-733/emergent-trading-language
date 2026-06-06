"""Tests for the agent's communication bottleneck."""

import unittest

import torch

from agents.agent import Agent


class AgentArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = Agent(
            obs_dim=4,
            vocab_size=4,
            msg_length=1,
            n_resources=2,
        )
        self.message = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    def test_inventory_extraction_removes_utility(self):
        obs = torch.tensor([[2.0, 4.0, 0.9, 0.1]])

        inventory = self.agent.extract_inventory(obs)

        self.assertTrue(torch.equal(inventory, torch.tensor([[2.0, 4.0]])))

    def test_actor_cannot_use_utility_directly(self):
        high_first_utility = torch.tensor([[2.0, 4.0, 0.9, 0.1]])
        high_second_utility = torch.tensor([[2.0, 4.0, 0.1, 0.9]])

        offer_a, _ = self.agent.act(
            high_first_utility,
            self.message,
            deterministic=True,
        )
        offer_b, _ = self.agent.act(
            high_second_utility,
            self.message,
            deterministic=True,
        )

        self.assertTrue(torch.equal(offer_a, offer_b))

    def test_actor_requires_full_observation(self):
        with self.assertRaises(ValueError):
            self.agent.act(
                torch.tensor([[2.0, 4.0]]),
                self.message,
                deterministic=True,
            )


if __name__ == "__main__":
    unittest.main()
