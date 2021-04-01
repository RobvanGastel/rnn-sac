import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action -
                           F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


class RNNActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=False):

        obs = obs.permute(1, 0, 2)
        net_out = self.net(obs)
        net_out = net_out.permute(1, 0, 2)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # action_range = 10
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample(mu.shape)
        action_0 = torch.tanh(mu + std * z)
        action = 10 * action_0

        epsilon = 1e-6
        log_prob = Normal(mu, std).log_prob(mu + std * z)
        - torch.log(1. - action_0.pow(2) + epsilon) - np.log(10)
        log_prob = log_prob.sum(dim=-1, keepdims=True)
        return action, log_prob

        # Pre-squash distribution and sample
        # pi_distribution = Normal(mu, std)
        # if deterministic:
        #     # Only used for evaluating policy at test time.
        #     pi_action = mu
        # else:
        #     pi_action = pi_distribution.rsample()

        # if with_logprob:
        #     print(pi_distribution)
        #     print(pi_action.shape)
        #     # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        #     # NOTE: The correction formula is a little bit magic. To get an understanding
        #     # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        #     # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        #     # Try deriving it yourself as a (very difficult) exercise. :)
        #     logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        #     logp_pi -= (2*(np.log(2) - pi_action -
        #                 F.softplus(-2*pi_action))).sum(axis=1)
        # else:
        #     logp_pi = None

        # pi_action = torch.tanh(pi_action)
        # pi_action = self.act_limit * pi_action

        # return pi_action, logp_pi


class RNNQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim, activation):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.linear2 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        # self.linear4.apply(linear_weights_init)
        self.activation = activation

    def reset_lstm():
        return NotImplementedError

    def forward(self, obs, action, last_action):
        """ 
        obs shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        print(obs.shape)
        obs = obs.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # obs = torch.unsqueeze(obs, 0).permute(1, 0, 2)
        # action = torch.unsqueeze(action, 0).permute(1, 0, 2)
        # last_action = torch.unsqueeze(last_action, 0).permute(1, 0, 2)

        # branch 1
        fc_branch = torch.cat([obs, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([obs, last_action], -1)
        # linear layer for 3d input only applied on the last dim
        lstm_branch = self.activation(self.linear2(lstm_branch))
        # lstm_branch, lstm_hidden = self.lstm1(
        # lstm_branch, hidden_in)  # no activation after lstm
        lstm_branch, lstm_hidden = self.lstm1(
            lstm_branch)  # no activation after lstm

        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        # lstm_hidden is actually tuple: (hidden, cell)
        return x


class RNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        # TODO: Refactor policy Pi, RNNActor to be recurrent
        self.pi = RNNActor(obs_dim, act_dim, hidden_sizes,
                           activation, act_limit)
        self.q1 = RNNQFunction(
            obs_dim, act_dim, hidden_sizes[0], activation=nn.Tanh())
        self.q2 = RNNQFunction(
            obs_dim, act_dim, hidden_sizes[0], activation=nn.Tanh())

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
