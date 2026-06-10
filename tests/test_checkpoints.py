"""Tests for shared checkpoint selection."""

import os
import tempfile
import unittest

import torch

from utils.checkpoints import latest_checkpoint_path, load_latest_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def test_newest_checkpoint_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = os.path.join(directory, "give_based_team_reward.pt")
            state_path = os.path.join(directory, "give_based_team_reward_state.pt")
            torch.save({"name": "final"}, final_path)
            torch.save({"name": "state"}, state_path)
            os.utime(final_path, (1, 1))
            os.utime(state_path, (2, 2))

            selected = latest_checkpoint_path(directory)
            checkpoint, loaded_path = load_latest_checkpoint(directory)

        self.assertEqual(selected, state_path)
        self.assertEqual(loaded_path, state_path)
        self.assertEqual(checkpoint["name"], "state")

    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint_path(directory)


if __name__ == "__main__":
    unittest.main()
