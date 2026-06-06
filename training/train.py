"""Robust training loop for emergent communication trading agents."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent import Agent
from env.trading_env import TradingEnv

TRAINING_STATE_PATH = "checkpoints/give_based_team_reward_state.pt"
FINAL_CHECKPOINT_PATH = "checkpoints/give_based_team_reward.pt"


def save_training_state(
    path,
    agent_a,
    agent_b,
    optimizer,
    config,
    update,
    temperature,
    team_baseline,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "agent_a": agent_a.state_dict(),
            "agent_b": agent_b.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "update": update,
            "temperature": temperature,
            "team_baseline": team_baseline,
        },
        path,
    )


def load_training_state(path, agent_a, agent_b, optimizer, config):
    """Load a trusted local checkpoint created by this training script."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = state["config"]
    if saved_config.get("architecture") != config["architecture"]:
        raise RuntimeError("Checkpoint architecture does not match the current model.")
    agent_a.load_state_dict(state["agent_a"])
    agent_b.load_state_dict(state["agent_b"])
    optimizer.load_state_dict(state["optimizer"])
    return state


def optimal_joint_reward(env):
    value_gap = env.util_a - env.util_b
    b_to_a = float(np.dot(env.inv_b, np.maximum(value_gap, 0.0)))
    a_to_b = float(np.dot(env.inv_a, np.maximum(-value_gap, 0.0)))
    return max(b_to_a + a_to_b, 1e-8)


def trade_score(env, offer_a, offer_b):
    reward_a, reward_b, success = env.step(offer_a, offer_b)
    if not success:
        return 0, 0.0, 0, reward_a, reward_b
    team_reward = reward_a + reward_b
    efficiency = team_reward / optimal_joint_reward(env)
    useful = int(efficiency >= 0.60)
    return 1, float(efficiency), useful, reward_a, reward_b


def actions_to_offers(give_a_to_b, give_b_to_a):
    """Convert each actor's give-action into the environment's receive-offers."""
    offer_a = give_b_to_a
    offer_b = give_a_to_b
    return offer_a, offer_b


def sample_episode(env, agent_a, agent_b, temperature, progress=1.0):
    obs_a_np, obs_b_np = env.reset()
    obs_a = torch.tensor(obs_a_np, dtype=torch.float32).unsqueeze(0)
    obs_b = torch.tensor(obs_b_np, dtype=torch.float32).unsqueeze(0)

    msg_a, logp_speak_a = agent_a.speak(obs_a, temperature)
    msg_b, logp_speak_b = agent_b.speak(obs_b, temperature)
    probs_a = agent_a.message_probabilities(obs_a, temperature)
    probs_b = agent_b.message_probabilities(obs_b, temperature)
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
    logits_a_to_b = agent_a.offer_logits(
        obs_a,
        msg_b,
        temperature,
        offer_limit,
    )
    logits_b_to_a = agent_b.offer_logits(
        obs_b,
        msg_a,
        temperature,
        offer_limit,
    )
    preferred_by_b = obs_b[:, env.n_resources:].argmax(dim=-1)
    preferred_by_a = obs_a[:, env.n_resources:].argmax(dim=-1)
    target_a_to_b = torch.zeros(
        obs_a.shape[0],
        env.n_resources,
        dtype=torch.long,
    )
    target_b_to_a = torch.zeros(
        obs_b.shape[0],
        env.n_resources,
        dtype=torch.long,
    )
    target_a_to_b.scatter_(1, preferred_by_b.unsqueeze(-1), 1)
    target_b_to_a.scatter_(1, preferred_by_a.unsqueeze(-1), 1)
    auxiliary_a_to_b = F.cross_entropy(
        logits_a_to_b.reshape(-1, agent_a.max_offer + 1),
        target_a_to_b.reshape(-1),
    )
    auxiliary_b_to_a = F.cross_entropy(
        logits_b_to_a.reshape(-1, agent_b.max_offer + 1),
        target_b_to_a.reshape(-1),
    )

    give_a_to_b = give_a_to_b_t.squeeze(0).detach().cpu().numpy().astype(int)
    give_b_to_a = give_b_to_a_t.squeeze(0).detach().cpu().numpy().astype(int)
    offer_a, offer_b = actions_to_offers(give_a_to_b, give_b_to_a)
    valid, efficiency, useful, reward_a, reward_b = trade_score(env, offer_a, offer_b)
    team_reward = reward_a + reward_b
    average_give = float(give_a_to_b.sum()) + float(give_b_to_a.sum())

    return {
        "valid": valid,
        "efficiency": efficiency,
        "useful": useful,
        "reward_a": reward_a,
        "reward_b": reward_b,
        "team_reward": team_reward,
        "average_give": average_give,
        "logp_agent_a": logp_speak_a + logp_act_a,
        "logp_agent_b": logp_speak_b + logp_act_b,
        "probs_a": probs_a,
        "probs_b": probs_b,
        "auxiliary_a_to_b": auxiliary_a_to_b,
        "auxiliary_b_to_a": auxiliary_b_to_a,
    }


def evaluate(agent_a, agent_b, config, mode="normal", n_episodes=3000):
    env = TradingEnv(
        n_resources=config["n_resources"],
        max_inventory=config["max_inventory"],
    )
    valid_log, useful_log, efficiency_log = [], [], []
    reward_a_log, reward_b_log, team_reward_log, give_log = [], [], [], []

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
            valid, efficiency, useful, reward_a, reward_b = trade_score(
                env,
                offer_a,
                offer_b,
            )
            valid_log.append(valid)
            useful_log.append(useful)
            efficiency_log.append(efficiency)
            reward_a_log.append(reward_a)
            reward_b_log.append(reward_b)
            team_reward_log.append(reward_a + reward_b)
            give_log.append(float(give_a_to_b.sum()) + float(give_b_to_a.sum()))

    agent_a.train()
    agent_b.train()
    return {
        "valid": float(np.mean(valid_log)),
        "useful": float(np.mean(useful_log)),
        "efficiency": float(np.mean(efficiency_log)),
        "reward_a": float(np.mean(reward_a_log)),
        "reward_b": float(np.mean(reward_b_log)),
        "team_reward": float(np.mean(team_reward_log)),
        "average_give": float(np.mean(give_log)),
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
        valid, efficiency, useful, _, _ = trade_score(env, offer_a, offer_b)
        valid_log.append(valid)
        useful_log.append(useful)
        efficiency_log.append(efficiency)
    return {
        "valid": float(np.mean(valid_log)),
        "useful": float(np.mean(useful_log)),
        "efficiency": float(np.mean(efficiency_log)),
    }


def training_stalled(update, efficiency):
    """Return whether the update-2000 diagnostic should warn."""
    return update == 2000 and round(efficiency, 3) == 0


def train(resume=True, n_updates=25000, config_overrides=None):
    config = {
        "n_resources": 2,
        "max_inventory": 5,
        "max_offer": 5,
        "vocab_size": 4,
        "msg_length": 1,
        "hidden_dim": 96,
        "lr": 7e-4,
        "updates": n_updates,
        "batch_size": 64,
        "gumbel_temp": 1.2,
        "temp_anneal": 0.99985,
        "message_mi_weight": 0.10,
        "utility_aux_weight": 1.00,
        "architecture": "inventory_message_team_reward_v5",
    }
    if config_overrides:
        config.update(config_overrides)
    if config["updates"] <= 0:
        raise ValueError("updates must be a positive integer.")

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
    running_team_baseline = 0.0
    start_update = 1

    if resume and os.path.exists(TRAINING_STATE_PATH):
        state = load_training_state(
            TRAINING_STATE_PATH,
            agent_a,
            agent_b,
            opt,
            config,
        )
        start_update = int(state["update"]) + 1
        temperature = float(state["temperature"])
        running_team_baseline = float(state["team_baseline"])
        print(f"Resuming training from update {start_update}...")
    else:
        print("Training started...")

    print(
        f"Random baseline | valid={baseline['valid']:.1%} | "
        f"useful={baseline['useful']:.1%} | efficiency={baseline['efficiency']:.3f}"
    )
    print(
        f"{'Update':>8} | {'Valid':>8} | {'Useful':>8} | {'Efficiency':>10} | "
        f"{'No-msg':>8} | {'Lang adv':>8} | {'Avg give':>8} | {'Team reward':>11}"
    )
    print("-" * 101)

    completed_update = start_update - 1
    try:
        for update in range(start_update, config["updates"] + 1):
            progress = update / config["updates"]
            episodes = [
                sample_episode(env, agent_a, agent_b, temperature, progress)
                for _ in range(config["batch_size"])
            ]
            team_rewards = np.array(
                [ep["team_reward"] for ep in episodes],
                dtype=np.float32,
            )
            average_team_rewards = 0.5 * team_rewards
            running_team_baseline = (
                0.95 * running_team_baseline
                + 0.05 * float(average_team_rewards.mean())
            )
            advantages_agent_a = torch.tensor(
                average_team_rewards - running_team_baseline,
                dtype=torch.float32,
            ).view(-1, 1)
            advantages_agent_b = torch.tensor(
                average_team_rewards - running_team_baseline,
                dtype=torch.float32,
            ).view(-1, 1)

            logps_agent_a = torch.cat(
                [ep["logp_agent_a"] for ep in episodes],
                dim=0,
            )
            logps_agent_b = torch.cat(
                [ep["logp_agent_b"] for ep in episodes],
                dim=0,
            )
            loss_agent_a = -(advantages_agent_a * logps_agent_a).mean()
            loss_agent_b = -(advantages_agent_b * logps_agent_b).mean()
            probs_a = torch.cat([ep["probs_a"] for ep in episodes], dim=0)
            probs_b = torch.cat([ep["probs_b"] for ep in episodes], dim=0)
            marginal_a = probs_a.mean(dim=0)
            marginal_b = probs_b.mean(dim=0)
            marginal_entropy_a = -(marginal_a * torch.log(marginal_a + 1e-8)).sum()
            marginal_entropy_b = -(marginal_b * torch.log(marginal_b + 1e-8)).sum()
            conditional_entropy_a = -(
                probs_a * torch.log(probs_a + 1e-8)
            ).sum(dim=-1).mean()
            conditional_entropy_b = -(
                probs_b * torch.log(probs_b + 1e-8)
            ).sum(dim=-1).mean()
            message_information = (
                marginal_entropy_a
                + marginal_entropy_b
                - conditional_entropy_a
                - conditional_entropy_b
            )
            strict_weight = min(1.0, progress / 0.60)
            message_weight = config["message_mi_weight"] * (1.0 - 0.90 * strict_weight)
            auxiliary_loss = torch.stack(
                [
                    ep["auxiliary_a_to_b"] + ep["auxiliary_b_to_a"]
                    for ep in episodes
                ]
            ).mean()
            loss = loss_agent_a + loss_agent_b
            loss -= message_weight * message_information
            loss += (
                config["utility_aux_weight"]
                * (1.0 - strict_weight)
                * auxiliary_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent_a.parameters()) + list(agent_b.parameters()),
                1.0,
            )
            opt.step()
            completed_update = update

            temperature = max(0.25, temperature * config["temp_anneal"])

            if update % 500 == 0:
                normal = evaluate(
                    agent_a,
                    agent_b,
                    config,
                    mode="normal",
                    n_episodes=800,
                )
                no_msg = evaluate(agent_a, agent_b, config, mode="zero", n_episodes=800)
                language_advantage = normal["efficiency"] - no_msg["efficiency"]
                collapse = (
                    " ZERO-GIVE COLLAPSE" if normal["average_give"] < 0.10 else ""
                )
                print(
                    f"{update:>8} | {normal['valid']:>7.1%} | "
                    f"{normal['useful']:>7.1%} | {normal['efficiency']:>10.3f} | "
                    f"{no_msg['efficiency']:>8.3f} | {language_advantage:>8.3f} | "
                    f"{normal['average_give']:>8.3f} | "
                    f"{normal['team_reward']:>11.3f}{collapse}"
                )
                if training_stalled(update, normal["efficiency"]):
                    print("Warning: training is not learning; check reward signal.")
                save_training_state(
                    TRAINING_STATE_PATH,
                    agent_a,
                    agent_b,
                    opt,
                    config,
                    completed_update,
                    temperature,
                    running_team_baseline,
                )
                print(f"         Saved progress at update {completed_update}.")
    except KeyboardInterrupt:
        save_training_state(
            TRAINING_STATE_PATH,
            agent_a,
            agent_b,
            opt,
            config,
            completed_update,
            temperature,
            running_team_baseline,
        )
        print(f"\nTraining interrupted. Progress saved at update {completed_update}.")
        print("Run the same command again to resume.")
        return

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "agent_a": agent_a.state_dict(),
            "agent_b": agent_b.state_dict(),
            "config": config,
        },
        FINAL_CHECKPOINT_PATH,
    )

    normal = evaluate(agent_a, agent_b, config, mode="normal")
    zero = evaluate(agent_a, agent_b, config, mode="zero")
    random_msg = evaluate(agent_a, agent_b, config, mode="random")

    print("\nTraining complete! Agents saved to checkpoints/")
    print(f"Normal messages efficiency : {normal['efficiency']:.3f}")
    print(f"Zero-message efficiency    : {zero['efficiency']:.3f}")
    print(f"Random-message efficiency  : {random_msg['efficiency']:.3f}")
    print(f"Random baseline efficiency : {baseline['efficiency']:.3f}")
    print(f"Normal reward A            : {normal['reward_a']:.3f}")
    print(f"Normal reward B            : {normal['reward_b']:.3f}")
    print(f"Normal team reward         : {normal['team_reward']:.3f}")
    print(f"Normal average give        : {normal['average_give']:.3f}")
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

