import numpy as np
import torch
import random

import core


class ReplayBuffer:
    """A FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in batch.items()}


"""Replay buffer for agent with GRU network additionally storing previous
action, initial input hidden state and output hidden state of the GRU.
And each sample contains the whole episode instead of a single step.
'hidden_in' and 'hidden_out' are only the initial hidden state for each
episode, for GRU initialization.
"""


class ReplayBufferGRU:
    """A FIFO experience replay buffer for GRU policy SAC agents.
    """

    def __init__(self, size):
        self.capacity = size
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def store(self, last_act, obs, act, rew, next_obs, hid_in,
              hid_out, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            hid_in, hid_out, obs, act, last_act, rew, next_obs, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample_batch(self, batch_size=32):
        o_lst, a_lst, a2_lst, r_lst, o2_lst, hi_lst, \
            ho_lst, d_lst = [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        for sample in batch:
            h_in, h_out, obs, act, act2, reward, obs2, done = sample
            o_lst.append(obs)
            o2_lst.append(obs2)
            a_lst.append(act)
            a2_lst.append(act2)
            r_lst.append(reward)
            d_lst.append(done)

            # Hidden states dimensions
            hi_lst.append(h_in)  # shape (1, batch_size, hidden_size)
            ho_lst.append(h_out)  # shape (1, batch_size, hidden_size)

        # concatenate along the batch dim
        hi_lst = torch.cat(hi_lst, dim=-2)
        ho_lst = torch.cat(ho_lst, dim=-2)

        batch = dict(
            hid_in=hi_lst,
            hid_out=ho_lst,
            act2=a2_lst,
            obs=o_lst,
            obs2=o2_lst,
            act=a_lst,
            rew=r_lst,
            done=d_lst)

        return {k: torch.tensor(v, dtype=torch.float32).cuda()
                if type(v) != tuple else v
                for k, v in batch.items()}


class ReplayBufferLSTM:
    """A FIFO experience replay buffer for LSTM policy SAC agents.
    """

    def __init__(self, size):
        self.capacity = size
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def store(self, last_act, obs, act, rew, next_obs, hid_in,
              hid_out, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            hid_in, hid_out, obs, act, last_act, rew, next_obs, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample_batch(self, batch_size=32):
        o_lst, a_lst, a2_lst, r_lst, o2_lst, hi_lst, \
            ci_lst, ho_lst, co_lst, d_lst = [
            ], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        # TODO: Omit this for-loop by moving it to torch/np
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, \
                reward, next_state, done = sample
            o_lst.append(state)
            a_lst.append(action)
            a2_lst.append(last_action)
            r_lst.append(reward)
            o2_lst.append(next_state)
            d_lst.append(done)

            # Hidden state dimensions
            hi_lst.append(h_in)  # shape (1, batch_size, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)

        # concatenate along the batch dim
        hi_lst = torch.cat(hi_lst, dim=-2).detach()
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        batch = dict(
            hid_in=hidden_in,
            hid_out=hidden_out,
            act2=a2_lst,
            obs=o_lst,
            obs2=o2_lst,
            act=a_lst,
            rew=r_lst,
            done=d_lst)

        return {k: torch.tensor(v, dtype=torch.float32).cuda()
                if type(v) != tuple else v
                for k, v in batch.items()}
