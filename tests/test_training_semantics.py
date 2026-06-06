"""Tests for the interpretation of actor actions during trading."""

import unittest

import numpy as np

from training.train import actions_to_offers


class TrainingSemanticsTests(unittest.TestCase):
    def test_actor_actions_are_given_to_the_other_agent(self):
        give_a_to_b = np.array([1, 2])
        give_b_to_a = np.array([3, 4])

        offer_a, offer_b = actions_to_offers(give_a_to_b, give_b_to_a)

        self.assertTrue(np.array_equal(offer_a, give_b_to_a))
        self.assertTrue(np.array_equal(offer_b, give_a_to_b))


if __name__ == "__main__":
    unittest.main()
