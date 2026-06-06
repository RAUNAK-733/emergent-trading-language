"""Random baseline agent for the trading environment."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import TradingEnv


class RandomAgent:
    """Agent that proposes blind random resource offers."""

    def __init__(self, n_resources, max_inventory=5):
        self.n_resources = n_resources
        self.max_inventory = max_inventory

    def act(self):
        return np.random.randint(0, self.max_inventory + 1, size=self.n_resources)


def optimal_joint_reward(env):
    value_gap = env.util_a - env.util_b
    b_to_a = float(np.dot(env.inv_b, np.maximum(value_gap, 0.0)))
    a_to_b = float(np.dot(env.inv_a, np.maximum(-value_gap, 0.0)))
    return max(b_to_a + a_to_b, 1e-8)


def evaluate_baseline(n_episodes=2000, n_resources=2, max_inventory=5):
    env = TradingEnv(n_resources=n_resources, max_inventory=max_inventory)
    agent_a = RandomAgent(n_resources, max_inventory)
    agent_b = RandomAgent(n_resources, max_inventory)
    successes = 0
    useful = 0
    efficiencies = []
    for _ in range(n_episodes):
        env.reset()
        offer_a = agent_a.act()
        offer_b = agent_b.act()
        reward_a, reward_b, success = env.step(offer_a, offer_b)
        if success:
            successes += 1
            efficiency = (reward_a + reward_b) / optimal_joint_reward(env)
            useful += int(efficiency >= 0.60)
            efficiencies.append(efficiency)
        else:
            efficiencies.append(0.0)
    rate = successes / n_episodes
    useful_rate = useful / n_episodes
    efficiency = float(np.mean(efficiencies))
    print(f"Random baseline valid trade rate : {rate:.1%}")
    print(f"Random baseline useful trade rate: {useful_rate:.1%}")
    print(f"Random baseline efficiency       : {efficiency:.3f}")
    print("(Your trained agents must beat this significantly)")
    return rate, useful_rate, efficiency


if __name__ == "__main__":
    evaluate_baseline()
