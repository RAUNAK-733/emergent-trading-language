"""Trading environment for two-agent exchange experiments."""

import numpy as np


class TradingEnv:
    """Simple two-agent trading environment."""

    def __init__(self, n_resources=2, max_inventory=5):
        self.n_resources = n_resources
        self.max_inventory = max_inventory

    def reset(self):
        """Sample inventories and utilities for both agents."""
        self.inv_a = np.random.randint(
            1,
            self.max_inventory + 1,
            size=self.n_resources,
        )
        self.inv_b = np.random.randint(
            1,
            self.max_inventory + 1,
            size=self.n_resources,
        )
        alpha = np.ones(self.n_resources)
        self.util_a = np.random.dirichlet(alpha)
        self.util_b = np.random.dirichlet(alpha)
        return self.obs_a(), self.obs_b()

    def step(self, offer_a, offer_b):
        """Evaluate a proposed trade and return rewards plus success flag."""
        offer_a = np.asarray(offer_a)
        offer_b = np.asarray(offer_b)
        nonnegative = np.all(offer_a >= 0) and np.all(offer_b >= 0)
        affordable = np.all(offer_a <= self.inv_b) and np.all(offer_b <= self.inv_a)
        mutual = offer_a.sum() > 0 and offer_b.sum() > 0
        valid = bool(nonnegative and affordable and mutual)

        if valid:
            reward_a = float(
                np.dot(offer_a, self.util_a) - np.dot(offer_b, self.util_a)
            )
            reward_b = float(
                np.dot(offer_b, self.util_b) - np.dot(offer_a, self.util_b)
            )
            trade_success = reward_a > 0.0 and reward_b > 0.0
        else:
            reward_a = 0.0
            reward_b = 0.0
            trade_success = False

        if not trade_success:
            reward_a = 0.0
            reward_b = 0.0

        return reward_a, reward_b, trade_success

    def obs_a(self):
        return np.concatenate([self.inv_a, self.util_a])

    def obs_b(self):
        return np.concatenate([self.inv_b, self.util_b])


if __name__ == "__main__":
    env = TradingEnv()
    for i in range(5):
        obs_a, obs_b = env.reset()
        offer_a = np.random.randint(0, 3, size=2)
        offer_b = np.random.randint(0, 3, size=2)
        r_a, r_b, success = env.step(offer_a, offer_b)
        print(
            f"Episode {i}: trade={'OK' if success else 'FAIL'}, "
            f"rewards=({r_a:.2f}, {r_b:.2f})"
        )
