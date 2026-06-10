"""Tests for experiment status reporting."""

import os
import tempfile
import unittest

import torch

from agents.agent import Agent
from utils.status import experiment_status


class StatusTests(unittest.TestCase):
    def test_status_reads_checkpoint_and_log(self):
        agent = Agent(4, 4, 1, 2, hidden_dim=96, max_offer=5)
        config = {
            "n_resources": 2,
            "vocab_size": 4,
            "msg_length": 1,
            "hidden_dim": 96,
            "max_offer": 5,
            "architecture": "inventory_message_team_reward_v5",
            "updates": 1000,
            "seed": 7,
        }
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {
                    "agent_a": agent.state_dict(),
                    "agent_b": agent.state_dict(),
                    "config": config,
                    "update": 500,
                },
                os.path.join(directory, "give_based_team_reward_state.pt"),
            )

            status = experiment_status(directory)

        self.assertEqual(status["update"], 500)
        self.assertEqual(status["target_updates"], 1000)
        self.assertAlmostEqual(status["progress"], 0.5)
