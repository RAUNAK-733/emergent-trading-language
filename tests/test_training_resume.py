"""Tests for interruption-safe training checkpoints."""

import os
import tempfile
import unittest

import torch

from agents.agent import Agent
from training.train import (
    load_training_state,
    move_optimizer_to_device,
    restore_rng_state,
    save_training_state,
    set_seed,
)


class TrainingResumeTests(unittest.TestCase):
    def test_training_state_round_trip(self):
        config = {"architecture": "inventory_message_team_reward_v5"}
        agent_a = Agent(4, 4, 1, 2)
        agent_b = Agent(4, 4, 1, 2)
        optimizer = torch.optim.Adam(
            list(agent_a.parameters()) + list(agent_b.parameters()),
            lr=1e-3,
        )
        original = agent_a.speak_net[0].weight.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_state.pt")
            save_training_state(
                path,
                agent_a,
                agent_b,
                optimizer,
                config,
                update=500,
                temperature=0.9,
                team_baseline=0.1,
            )
            with torch.no_grad():
                agent_a.speak_net[0].weight.zero_()

            state = load_training_state(path, agent_a, agent_b, optimizer, config)

        self.assertEqual(state["update"], 500)
        self.assertEqual(state["temperature"], 0.9)
        self.assertEqual(state["team_baseline"], 0.1)
        self.assertTrue(torch.equal(agent_a.speak_net[0].weight, original))

    def test_optimizer_state_can_move_to_model_device(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()

        move_optimizer_to_device(optimizer, parameter.device)

        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device, parameter.device)

    def test_rng_state_restores_numpy_and_torch(self):
        set_seed(17)
        numpy_state = __import__("numpy").random.get_state()
        torch_state = torch.get_rng_state()
        expected_numpy = __import__("numpy").random.random()
        expected_torch = torch.rand(1)

        restore_rng_state(
            {
                "numpy_rng_state": numpy_state,
                "torch_rng_state": torch_state,
                "cuda_rng_state": None,
            }
        )

        self.assertEqual(__import__("numpy").random.random(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))


if __name__ == "__main__":
    unittest.main()
