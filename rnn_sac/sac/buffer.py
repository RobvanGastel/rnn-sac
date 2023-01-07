import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class EpisodicBuffer:
    """
    A buffer for storing trajectories experienced by a SAC agent
    interacting with the environment
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        size,
        hidden_size,
        device,
        use_sac=False,
        use_exploration_sampling=False,
    ):

        # Pick sampling episodes or time-steps
        self.exploration_batch = np.array([])
        self.exploitation_batch = np.array([])

        self.use_sac = use_sac

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.size = size

        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        # RL^2 variables
        self.prev_act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.prev_rew_buf = np.zeros(size, dtype=np.float32)

        if use_sac:
            self.next_hxs_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.hxs_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.use_exploration_sampling = use_exploration_sampling
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, next_obs, act, rew, done, prev_act, prev_rew, hidden):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.prev_act_buf[self.ptr] = prev_act
        self.prev_rew_buf[self.ptr] = prev_rew

        if self.use_sac:
            hid, next_hid = hidden
            self.hxs_buf[self.ptr] = hid
            self.next_hxs_buf[self.ptr] = next_hid
        else:
            self.hxs_buf[self.ptr] = hidden

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def finish_path_sac(self):

        path_slice = slice(self.path_start_idx, self.ptr)

        # Exploitation batch
        data = dict(
            obs=self.obs_buf[path_slice],
            obs2=self.next_obs_buf[path_slice],
            act=self.act_buf[path_slice],
            rew=self.rew_buf[path_slice],
            done=self.done_buf[path_slice],
            prev_act=self.prev_act_buf[path_slice],
            prev_rew=self.prev_rew_buf[path_slice],
            hid=self.hxs_buf[self.path_start_idx],
            hid_out=self.next_hxs_buf[self.path_start_idx],
        )
        data = {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in data.items()
        }
        self.exploitation_batch = np.append(self.exploitation_batch, data)

        if self.use_exploration_sampling:
            # Exploration batch
            # set the return of the episode to 0
            rews_zero = np.zeros_like(self.rew_buf[path_slice])

            data = dict(
                obs=self.obs_buf[path_slice],
                obs2=self.next_obs_buf[path_slice],
                act=self.act_buf[path_slice],
                rew=rews_zero,
                done=self.done_buf[path_slice],
                prev_act=self.prev_act_buf[path_slice],
                prev_rew=self.prev_rew_buf[path_slice],
                hid=self.hxs_buf[self.path_start_idx],
                hid_out=self.next_hxs_buf[self.path_start_idx],
            )
            data = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in data.items()
            }
            self.exploration_batch = np.append(self.exploration_batch, data)

        self.path_start_idx = self.ptr

    def finish_path(self):

        if self.use_sac:
            self.finish_path_sac()
            return

        # Exploitation batch
        path_slice = slice(self.path_start_idx, self.ptr)
        rews_zero = np.zeros_like(self.rew_buf[path_slice])

        data = dict(
            obs=self.obs_buf[path_slice],
            obs2=self.next_obs_buf[path_slice],
            act=self.act_buf[path_slice],
            rew=self.rew_buf[path_slice],
            done=self.done_buf[path_slice],
            prev_act=self.prev_act_buf[path_slice],
            prev_rew=self.prev_rew_buf[path_slice],
            hidden=self.hxs_buf[self.path_start_idx],
        )

        data = {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in data.items()
        }
        self.exploitation_batch = np.append(self.exploitation_batch, data)

        if self.use_exploration_sampling:
            # Exploration batch
            # set the return of the episode to 0
            rews_zero = np.zeros_like(self.rew_buf[path_slice])

            data = dict(
                obs=self.obs_buf[path_slice],
                obs2=self.next_obs_buf[path_slice],
                act=self.act_buf[path_slice],
                rew=rews_zero,
                done=self.done_buf[path_slice],
                prev_act=self.prev_act_buf[path_slice],
                prev_rew=self.prev_rew_buf[path_slice],
                hidden=self.hxs_buf[self.path_start_idx],
            )
            data = {
                k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in data.items()
            }
            self.exploration_batch = np.append(self.exploration_batch, data)

        self.path_start_idx = self.ptr

    def get(self, batch_size=None, p_exploration=0.3):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # buffer has to be full before you can get
        # assert self.ptr == self.max_size

        if self.use_exploration_sampling:
            # use_exploration_sampling
            if batch_size is not None:
                k = batch_size
                p = int(k * p_exploration)

                # p explore-rollouts
                explore_idx = np.random.choice(
                    np.arange(len(self.exploration_batch)), p, replace=False
                )

                b_mask = np.ones_like(self.exploration_batch, dtype=bool)
                # Exclude p explore rollouts
                b_mask[explore_idx] = False

                # k-p exploit-rollouts
                explore = self.exploration_batch[explore_idx]
                exploit = self.exploitation_batch[b_mask]

                exploit = np.random.choice(
                    self.exploitation_batch, k - p, replace=False
                )

                batch = [*explore, *exploit]
                np.random.shuffle(batch)
                return batch

            k = len(self.exploitation_batch)
            p = int(k * p_exploration)

            # p explore-rollouts
            explore_idx = np.random.choice(
                np.arange(len(self.exploration_batch)), p, replace=False
            )

            b_mask = np.ones_like(self.exploration_batch, dtype=bool)
            # Exclude p explore rollouts
            b_mask[explore_idx] = False

            # k-p exploit-rollouts
            explore = self.exploration_batch[explore_idx]
            exploit = self.exploitation_batch[b_mask]

            batch = [*explore, *exploit]
            np.random.shuffle(batch)
            return batch

        if batch_size is not None:
            np.random.shuffle(self.exploitation_batch)
            return np.random.choice(self.exploitation_batch, batch_size)

        np.random.shuffle(self.exploitation_batch)
        return self.exploitation_batch

    def reset(self):
        self.obs_buf = np.zeros_like(self.obs_buf)
        self.next_obs_buf = np.zeros_like(self.next_obs_buf)
        self.act_buf = np.zeros_like(self.act_buf)
        self.rew_buf = np.zeros_like(self.rew_buf)
        self.done_buf = np.zeros_like(self.done_buf)
        self.prev_act_buf = np.zeros_like(self.prev_act_buf)
        self.prev_rew_buf = np.zeros_like(self.prev_rew_buf)

        if self.use_sac:
            self.next_hxs_buf = np.zeros_like(self.next_hxs_buf)
        self.hxs_buf = np.zeros_like(self.hxs_buf)

        self.exploration_batch = np.array([])
        self.exploitation_batch = np.array([])
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size
