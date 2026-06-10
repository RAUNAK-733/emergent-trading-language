"""Tests for strict communication controls."""

import unittest

import torch

from analysis.verify import message_control


class VerifyControlTests(unittest.TestCase):
    def setUp(self):
        self.config = {"vocab_size": 4, "msg_length": 1}
        self.message = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    def test_zero_control_preserves_shape_and_device(self):
        controlled = message_control(self.message, self.config, "zero")

        self.assertEqual(controlled.shape, self.message.shape)
        self.assertEqual(controlled.device, self.message.device)
        self.assertEqual(float(controlled.sum()), 0.0)

    def test_random_control_preserves_shape_and_device(self):
        controlled = message_control(self.message, self.config, "random")

        self.assertEqual(controlled.shape, self.message.shape)
        self.assertEqual(controlled.device, self.message.device)
        self.assertEqual(float(controlled.sum()), 1.0)

    def test_unknown_control_raises(self):
        with self.assertRaises(ValueError):
            message_control(self.message, self.config, "unknown")


if __name__ == "__main__":
    unittest.main()
