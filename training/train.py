"""Robust training loop for emergent communication trading agents."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent import Agent
from env.trading_env import TradingEnv


def optimal_joint_reward(env):
    value_gap = env.util_a - env.util_b
    b_to_a = float(np.dot(env.inv_b, np.maximum(value_gap, 0.0)))
    a_to_b = float(np.dot(env.inv_a, np.maximum(-value_gap, 0.0)))
    return max(b_to_a + a_to_b, 1e-8)


def trade_score(env, offer_a, offer_b):
    reward_a, reward_b, success = env.step(offer_a, offer_b)
    if not success:
        return 0, 0.0, 0.0
    joint_reward = reward_a + reward_b
    efficiency = joint_reward / optimal_joint_reward(env)
    useful = int(efficiency >= 0.60)
    return 1, float(efficiency), useful


def actions_to_offers(give_a_to_b, give_b_to_a):
    """Convert each actor's give-action into the environment's receive-offers."""
    offer_a = give_b_to_a
    offer_b = give_a_to_b
    return offer_a, offer_b


def shaped_objective(env, offer_a, offer_b, success, efficiency):
    """Dense learning signal; final evaluation still uses strict success."""
    overflow_a = np.maximum(offer_a - env.inv_b, 0).sum()
    overflow_b = np.maximum(offer_b - env.inv_a, 0).sum()
    overflow = (overflow_a + overflow_b) / (
        2 * env.n_resources * env.max_inventory
    )

    gain_a = float(np.dot(offer_a - offer_b, env.util_a))
    gain_b = float(np.dot(offer_b - offer_a, env.util_b))
    scale = optimal_joint_reward(env)
    welfare = (gain_a + gain_b) / scale
    fairness = min(gain_a, gain_b) / scale
    empty_penalty = 0.10 if offer_a.sum() == 0 or offer_b.sum() == 0 else 0.0

    objective = 0.45 * welfare + 0.75 * fairness - 0.60 * overflow - empty_penalty
    if success:
        objective += 0.25 + 0.50 * efficiency
    return float(np.clip(objective, -1.0, 1.5))


def curriculum_objectives(env, give_a_to_b, give_b_to_a, strict_objective, progress):
    """Train each message-action pair, then move to the strict team objective."""
    scale = env.n_resources * env.max_inventory
    overflow_a = np.maximum(give_a_to_b - env.inv_a, 0).sum() / scale
    overflow_b = np.maximum(give_b_to_a - env.inv_b, 0).sum() / scale
    size_a = float(give_a_to_b.sum())
    size_b = float(give_b_to_a.sum())
    alignment_a = float(np.dot(give_a_to_b, env.util_b)) / max(size_a, 1.0)
    alignment_b = float(np.dot(give_b_to_a, env.util_a)) / max(size_b, 1.0)

    excess_a = max(size_a - 1.0, 0.0) / scale
    excess_b = max(size_b - 1.0, 0.0) / scale
    empty_a = 0.25 if size_a == 0 else 0.0
    empty_b = 0.25 if size_b == 0 else 0.0
    bootstrap_a_to_b = alignment_a - 0.80 * overflow_a - 0.50 * excess_a - empty_a
    bootstrap_b_to_a = alignment_b - 0.80 * overflow_b - 0.50 * excess_b - empty_b

    strict_weight = min(1.0, progress / 0.60)
    objective_a_to_b = (1.0 - strict_weight) * bootstrap_a_to_b
    objective_b_to_a = (1.0 - strict_weight) * bootstrap_b_to_a
    objective_a_to_b += strict_weight * strict_objective
    objective_b_to_a += strict_weight * strict_objective
    return (
        float(np.clip(objective_a_to_b, -1.0, 1.5)),
        float(np.clip(objective_b_to_a, -1.0, 1.5)),
    )


def sample_episode(env, agent_a, agent_b, temperature, progress=1.0):
    obs_a_np, obs_b_np = env.reset()
    obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
    obs_b = torch.tensor(obs_b_np, dtype=torch.float32).unsqueeze(0)

    msg_a, logp_speak_a = agent_a.speak(obs_a, temperature)
    msg_b, logp_speak_b = agent_b.speak(obs_b, temperature)
    offer_limit = min(
        agent_a.max_offer,
        1 + int(progress * agent_a.max_offer),
    )
    give_a_to_b_t, logp_act_a = agent_a.act(
        obs_a,
        msg_b,
        temperature,
        offer_limit=offer_limit,
    )
    give_b_to_a_t, logp_act_b = agent_b.act(
        obs_b,
        msg_a,
        temperature,
        offer_limit=offer_limit,
    )

    give_a_to_b = give_a_to_b_t.squeeze(0).detach().cpu().numpy().astype(int)
    give_b_to_a = give_b_to_a_t.squeeze(0).detach().cpu().numpy().astype(int)
    offer_a, offer_b = actions_to_offers(give_a_to_b, give_b_to_a)
    valid, efficiency, useful = trade_score(env, offer_a, offer_b)

    strict_objective = shaped_objective(env, offer_a, offer_b, valid, efficiency)
    objective_a_to_b, objective_b_to_a = curriculum_objectives(
        env,
        give_a_to_b,
        give_b_to_a,
        strict_objective,
        progress,
    )

    return {
        "objective": 0.5 * (objective_a_to_b + objective_b_to_a),
        "objective_a_to_b": objective_a_to_b,
        "objective_b_to_a": objective_b_to_a,
        "valid": valid,
        "efficiency": efficiency,
        "useful": useful,
        "logp_a_to_b": logp_speak_b + logp_act_a,
        "logp_b_to_a": logp_speak_a + logp_act_b,
    }


def evaluate(agent_a, agent_b, config, mode="normal", n_episodes=3000):
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    valid_log, useful_log, efficiency_log = [], [], []

    agent_a.eval()
    agent_b.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs_a_np, obs_b_np = env.reset()
            obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
            obs_b = torch.tensor(obs_b_np, dtype=torch.float32).unsqueeze(0)

            msg_a, _ = agent_a.speak(obs_a, deterministic=True)
            msg_b, _ = agent_b.speak(obs_b, deterministic=True)

            if mode == "zero":
                msg_a = torch.zeros_like(msg_a)
                msg_b = torch.zeros_like(msg_b)
            elif mode == "random":
                idx_a = torch.randint(config["vocab_size"], (1, config["msg_length"]))
                idx_b = torch.randint(config["vocab_size"], (1, config["msg_length"]))
                msg_a = torch.nn.functional.one_hot(
                    idx_a,
                    num_classes=config["vocab_size"],
                ).float()
                msg_b = torch.nn.functional.one_hot(
                    idx_b,
                    num_classes=config["vocab_size"],
                ).float()
            elif mode != "normal":
                raise ValueError(f"Unknown evaluation mode: {mode}")

            give_a_to_b_t, _ = agent_a.act(obs_a, msg_b, deterministic=True)
            give_b_to_a_t, _ = agent_b.act(obs_b, msg_a, deterministic=True)
            give_a_to_b = give_a_to_b_t.squeeze(0).cpu().numpy().astype(int)
            give_b_to_a = give_b_to_a_t.squeeze(0).cpu().numpy().astype(int)
            offer_a, offer_b = actions_to_offers(give_a_to_b, give_b_to_a)
            valid, efficiency, useful = trade_score(env, offer_a, offer_b)
            valid_log.append(valid)
            useful_log.append(useful)
            efficiency_log.append(efficiency)

    agent_a.train()
    agent_b.train()
    return {
        "valid": float(np.mean(valid_log)),
        "useful": float(np.mean(useful_log)),
        "efficiency": float(np.mean(efficiency_log)),
    }


def random_baseline(config, n_episodes=5000):
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    valid_log, useful_log, efficiency_log = [], [], []
    for _ in range(n_episodes):
        env.reset()
        offer_a = np.random.randint(
            0,
            config["max_offer"] + 1,
            size=config["n_resources"],
        )
        offer_b = np.random.randint(
            0,
            config["max_offer"] + 1,
            size=config["n_resources"],
        )
        valid, efficiency, useful = trade_score(env, offer_a, offer_b)
        valid_log.append(valid)
        useful_log.append(useful)
        efficiency_log.append(efficiency)
    return {
        "valid": float(np.mean(valid_log)),
        "useful": float(np.mean(useful_log)),
        "efficiency": float(np.mean(efficiency_log)),
    }


def train():
    config = {
        "n_resources": 2,
        "max_inventory": 5,
        "max_offer": 5,
        "vocab_size": 4,
        "msg_length": 1,
        "hidden_dim": 96,
        "lr": 7e-4,
        "updates": 12000,
        "batch_size": 64,
        "gumbel_temp": 1.2,
        "temp_anneal": 0.99985,
        "architecture": "inventory_message_give_v2",
    }

    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    obs_dim = 2 * config["n_resources"]

    agent_a = Agent(
        obs_dim,
        config["vocab_size"],
        config["msg_length"],
        config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )
    agent_b = Agent(
        obs_dim,
        config["vocab_size"],
        config["msg_length"],
        config["n_resources"],
        hidden_dim=config["hidden_dim"],
        max_offer=config["max_offer"],
    )

    opt = torch.optim.Adam(
        list(agent_a.parameters()) + list(agent_b.parameters()),
        lr=config["lr"],
    )
    baseline = random_baseline(config)
    temperature = config["gumbel_temp"]
    running_baseline_a_to_b = 0.0
    running_baseline_b_to_a = 0.0

    print("Training started...")
    print(
        f"Random baseline | valid={baseline['valid']:.1%} | "
        f"useful={baseline['useful']:.1%} | efficiency={baseline['efficiency']:.3f}"
    )
    print(
        f"{'Update':>8} | {'Valid':>8} | {'Useful':>8} | "
        f"{'Efficiency':>10} | {'No-msg eff':>10} | {'Offer':>5} | {'Strict':>6}"
    )
    print("-" * 78)

    for update in range(1, config["updates"] + 1):
        progress = update / config["updates"]
        episodes = [
            sample_episode(env, agent_a, agent_b, temperature, progress)
            for _ in range(config["batch_size"])
        ]
        objectives_a_to_b = np.array(
            [ep["objective_a_to_b"] for ep in episodes],
            dtype=np.float32,
        )
        objectives_b_to_a = np.array(
            [ep["objective_b_to_a"] for ep in episodes],
            dtype=np.float32,
        )
        running_baseline_a_to_b = (
            0.95 * running_baseline_a_to_b
            + 0.05 * float(objectives_a_to_b.mean())
        )
        running_baseline_b_to_a = (
            0.95 * running_baseline_b_to_a
            + 0.05 * float(objectives_b_to_a.mean())
        )
        advantages_a_to_b = torch.tensor(
            objectives_a_to_b - running_baseline_a_to_b,
            dtype=torch.float32,
        ).view(-1, 1)
        advantages_b_to_a = torch.tensor(
            objectives_b_to_a - running_baseline_b_to_a,
            dtype=torch.float32,
        ).view(-1, 1)

        logps_a_to_b = torch.cat(
            [ep["logp_a_to_b"] for ep in episodes],
            dim=0,
        )
        logps_b_to_a = torch.cat(
            [ep["logp_b_to_a"] for ep in episodes],
            dim=0,
        )
        loss_a_to_b = -(advantages_a_to_b * logps_a_to_b).mean()
        loss_b_to_a = -(advantages_b_to_a * logps_b_to_a).mean()
        loss = loss_a_to_b + loss_b_to_a

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent_a.parameters()) + list(agent_b.parameters()),
            1.0,
        )
        opt.step()

        temperature = max(0.25, temperature * config["temp_anneal"])

        if update % 500 == 0:
            valid = np.mean([ep["valid"] for ep in episodes])
            useful = np.mean([ep["useful"] for ep in episodes])
            efficiency = np.mean([ep["efficiency"] for ep in episodes])
            no_msg = evaluate(agent_a, agent_b, config, mode="zero", n_episodes=800)
            print(
                f"{update:>8} | {valid:>7.1%} | {useful:>7.1%} | "
                f"{efficiency:>10.3f} | {no_msg['efficiency']:>10.3f} | "
                f"{min(config['max_offer'], 1 + int(progress * config['max_offer'])):>5} | "
                f"{min(1.0, progress / 0.60):>6.2f}"
            )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent_a.state_dict(), "checkpoints/agent_a.pt")
    torch.save(agent_b.state_dict(), "checkpoints/agent_b.pt")
    torch.save(config, "checkpoints/config.pt")

    normal = evaluate(agent_a, agent_b, config, mode="normal")
    zero = evaluate(agent_a, agent_b, config, mode="zero")
    random_msg = evaluate(agent_a, agent_b, config, mode="random")

    print("\nTraining complete! Agents saved to checkpoints/")
    print(f"Normal messages efficiency : {normal['efficiency']:.3f}")
    print(f"Zero-message efficiency    : {zero['efficiency']:.3f}")
    print(f"Random-message efficiency  : {random_msg['efficiency']:.3f}")
    print(f"Random baseline efficiency : {baseline['efficiency']:.3f}")
    zero_gain = normal["efficiency"] - zero["efficiency"]
    random_gain = normal["efficiency"] - random_msg["efficiency"]
    print(f"Advantage over zero message: {zero_gain:.3f}")
    print(f"Advantage over random msg  : {random_gain:.3f}")
    print(
        "Result                     : "
        f"{'STRONGER EVIDENCE' if normal['efficiency'] > max(zero['efficiency'], random_msg['efficiency'], baseline['efficiency']) + 0.05 else 'needs more training/complexity'}"
    )


if __name__ == "__main__":
    train()
