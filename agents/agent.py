"""Neural agent for emergent communication trading experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent(nn.Module):
    """Agent with stochastic message and offer policies."""

    def __init__(
        self,
        obs_dim,
        vocab_size,
        msg_length,
        n_resources,
        hidden_dim=64,
        max_offer=5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        self.n_resources = n_resources
        self.max_offer = max_offer

        self.speak_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size * msg_length),
        )

        self.act_net = nn.Sequential(
            nn.Linear(n_resources + msg_length * vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_resources * (max_offer + 1)),
        )

    def extract_inventory(self, obs_full):
        """Extract the inventory portion from a full observation."""
        if obs_full.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected full observation with {self.obs_dim} values, "
                f"received {obs_full.shape[-1]}."
            )
        return obs_full[:, : self.n_resources]

    def speak(self, obs_full, temperature=1.0, deterministic=False):
        logits = self.speak_net(obs_full)
        logits = logits.view(-1, self.msg_length, self.vocab_size)
        scaled_logits = logits / max(temperature, 1e-6)

        if deterministic:
            tokens = scaled_logits.argmax(dim=-1)
            log_prob = torch.zeros(obs_full.shape[0], 1, device=obs_full.device)
            message = F.one_hot(tokens, num_classes=self.vocab_size).float()
        else:
            message = F.gumbel_softmax(
                scaled_logits,
                tau=1.0,
                hard=True,
                dim=-1,
            )
            tokens = message.argmax(dim=-1)
            log_prob = torch.log_softmax(scaled_logits, dim=-1)
            log_prob = log_prob.gather(-1, tokens.unsqueeze(-1))
            log_prob = log_prob.sum(dim=-2)

        return message, log_prob

    def act(
        self,
        obs_full,
        received_msg,
        temperature=1.0,
        deterministic=False,
        offer_limit=None,
    ):
        inventory = self.extract_inventory(obs_full)
        msg_flat = received_msg.reshape(obs_full.shape[0], -1)
        x = torch.cat([inventory, msg_flat], dim=-1)
        logits = self.act_net(x)
        logits = logits.view(-1, self.n_resources, self.max_offer + 1)
        scaled_logits = logits / max(temperature, 1e-6)
        if offer_limit is not None:
            if not 0 <= offer_limit <= self.max_offer:
                raise ValueError(
                    f"offer_limit must be between 0 and {self.max_offer}."
                )
            values = torch.arange(self.max_offer + 1, device=scaled_logits.device)
            scaled_logits = scaled_logits.masked_fill(values > offer_limit, -1e9)

        if deterministic:
            offer = scaled_logits.argmax(dim=-1)
            log_prob = torch.zeros(obs_full.shape[0], 1, device=obs_full.device)
        else:
            dist = Categorical(logits=scaled_logits)
            offer = dist.sample()
            log_prob = dist.log_prob(offer).sum(dim=-1, keepdim=True)

        return offer.float(), log_prob


if __name__ == "__main__":
    agent_a = Agent(obs_dim=4, vocab_size=4, msg_length=1, n_resources=2)
    agent_b = Agent(obs_dim=4, vocab_size=4, msg_length=1, n_resources=2)

    obs = torch.randn(1, 4)

    msg_a, logp_speak_a = agent_a.speak(obs)
    msg_b, logp_speak_b = agent_b.speak(obs)

    action_a, logp_act_a = agent_a.act(obs, msg_b)
    action_b, logp_act_b = agent_b.act(obs, msg_a)

    print("msg_a shape     :", msg_a.shape)
    print("action_a shape  :", action_a.shape)
    print("action_a values :", action_a)
    print("Agent brains wired correctly!")
