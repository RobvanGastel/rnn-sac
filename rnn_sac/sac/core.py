import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


"""Discrete Categorical Policy"""


class CategoricalPolicy(nn.Module):
    def __init__(self, act_dim, hidden_size=64, activation=nn.ReLU()):
        super().__init__()

        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, act_dim)

    def act(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        action_logits = self.policy(mem)

        # Greedy action selection
        return torch.argmax(action_logits, dim=-1)

    def sample(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        action_logits = self.policy(mem)

        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class GaussianPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-8

    def __init__(self, act_dim, act_limit, hidden_size=64, activation=nn.ReLU()):
        super().__init__()
        # TODO:
        # act_limit = 2
        self.act_limit = act_limit

        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.mu_layer = nn.Linear(hidden_size, act_dim)
        self.log_std_layer = nn.Linear(hidden_size, act_dim)

    def forward(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))

        mu = self.mu_layer(mem)
        log_std = self.log_std_layer(mem)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        std = log_std.exp()
        return mu, std

    def act(self, memory_emb):
        # Deterministic version
        mu, std = self.forward(memory_emb)

        pi_dist = Normal(mu, std)

        # Deterministic action selection
        pi_action = mu

        log_action_probs = pi_dist.log_prob(pi_action).sum(axis=-1)
        log_action_probs -= (
            2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
        ).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        actions = self.act_limit * pi_action

        return actions, None, log_action_probs

    def sample(self, memory_emb):
        mu, std = self.forward(memory_emb)
        print(mu.shape, std.shape)
        pi_dist = Normal(mu, std)
        pi_action = pi_dist.rsample()

        log_action_probs = pi_dist.log_prob(pi_action).sum(axis=-1)
        log_action_probs -= (
            2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
        ).sum(axis=1)

        pi_action = torch.tanh(pi_action)
        actions = self.act_limit * pi_action
        print(actions.shape)

        # log_prob => log_action_probs
        return None, actions, log_action_probs


class Memory(nn.Module):
    def __init__(self, obs_dim, act_dim, device, hidden_size=64, activation=nn.ReLU()):
        super().__init__()

        self.activation = activation
        self.act_dim = act_dim
        self.device = device

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        # +1 for the reward
        self.gru = nn.GRU(hidden_size + act_dim + 1, hidden_size, batch_first=True)

    def _one_hot(self, act):
        print(self.act_dim, act)
        return torch.eye(self.act_dim)[act.long(), :].to(self.device)

    def forward(self, obs, prev_act, prev_rew, hid_in, training=False):

        # act_enc = self._one_hot(prev_act)
        act_enc = torch.tensor(prev_act)
        obs_enc = self.activation(self.linear1(obs))

        gru_input = torch.cat([obs_enc, act_enc, prev_rew], dim=-1,)

        # Input rnn: (batch size, sequence length, features)
        if training:
            gru_input = gru_input.unsqueeze(0)
            gru_out, hid_out = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(0)
        else:
            gru_input = gru_input.unsqueeze(1)
            gru_out, hid_out = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(1)

        return gru_out, hid_out


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU()):
        super().__init__()

        self.activation = activation

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.q = nn.Linear(hidden_size, act_dim)

    def forward(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        q = self.q(mem)
        return q


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_size=[256, 256],
        activation=nn.ReLU(),
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # act_dim = action_space.n
        act_dim = action_space.shape[0]
        self.device = device

        act_limit = action_space.high[0]

        self.memory = Memory(
            obs_dim, act_dim, device, hidden_size[0], activation=activation
        )
        # self.pi = CategoricalPolicy(
        #     act_dim, hidden_size=hidden_size[0], activation=activation)
        self.pi = GaussianPolicy(
            act_dim, act_limit, hidden_size[0], activation=activation
        )

        self.q1 = QNetwork(
            obs_dim, act_dim, hidden_size=hidden_size[0], activation=activation
        )
        self.q2 = QNetwork(
            obs_dim, act_dim, hidden_size=hidden_size[0], activation=activation
        )

    def act(self, obs, prev_act, prev_rew, hid_in, training=False):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(obs, prev_act, prev_rew, hid_in, training)
            action = self.pi.act(mem_emb)

        return action.item(), hid_out

    def explore(self, obs, prev_act, prev_rew, hid_in, training=False):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(obs, prev_act, prev_rew, hid_in, training)
            print(mem_emb.shape)
            _, action, _ = self.pi.sample(mem_emb)
        print(action)
        return action.item(), hid_out
