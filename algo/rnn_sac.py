from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import random
import time
import algo.core as core
from algo.utils.logx import EpochLogger

"""Original implementation by OpenAI SpinningUp
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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


class ReplayBufferLSTM:
    """
    A simple FIFO experience replay buffer for meta-SAC agents.
    From origin:
    Replay buffer for agent with LSTM network additionally storing previous
    action, initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each
    episode, for LSTM initialization.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim, size, max_ep_len):
        # self.hid_in_buf = np.zeros((size, 2, hidden_dim), dtype=np.float32)
        # self.hid_out_buf = np.zeros((size, 2, hidden_dim), dtype=np.float32)

        # # TODO: Careful as the obs_dim is assumed to be 1D
        # self.obs_buf = np.zeros((
        #     size, max_ep_len, obs_dim[0]), dtype=np.float32)
        # self.obs2_buf = np.zeros((
        #     size, max_ep_len, obs_dim[0]), dtype=np.float32)
        # self.act_buf = np.zeros((
        #     size, max_ep_len, act_dim), dtype=np.float32)
        # self.act2_buf = np.zeros((
        #     size, max_ep_len, act_dim), dtype=np.float32)
        # self.rew_buf = np.zeros((
        #     size, max_ep_len), dtype=np.float32)
        # self.done_buf = np.zeros((
        #     size, max_ep_len), dtype=np.float32)

        # self.ptr, self.size, self.max_size = 0, 0, size
        self.capacity = size
        self.buffer = []
        self.position = 0

    def store(self, last_act, obs, act, rew,
              next_obs, hid_in, hid_out, done):
        if len(self.buffer) < self.capacity:
                self.buffer.append(None)
        self.buffer[self.position] = (
            hid_in, hid_out, obs, act, last_act, rew, next_obs, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

        # self.hid_in_buf[self.ptr] = hid_in
        # self.hid_out_buf[self.ptr] = hid_out

        # self.obs_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        # self.act_buf[self.ptr] = act
        # self.act2_buf[self.ptr] = last_act
        # self.rew_buf[self.ptr] = rew
        # self.done_buf[self.ptr] = done
        # self.ptr = (self.ptr+1) % self.max_size
        # self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        # idxs = np.random.randint(0, self.size, size=batch_size)

        # batch = dict(
        #     hid_in=self.hid_in_buf[idxs],
        #     hid_out=self.hid_out_buf[idxs],
        #     act2=self.act2_buf[idxs],
        #     obs=self.obs_buf[idxs],
        #     obs2=self.obs2_buf[idxs],
        #     act=self.act_buf[idxs],
        #     rew=self.rew_buf[idxs],
        #     done=self.done_buf[idxs])
        # return {k: torch.as_tensor(v, dtype=torch.float32)
        #         if type(v) != tuple else v for k, v in batch.items()}
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, \
            ci_lst, ho_lst, co_lst, d_lst = [
        ], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, \
                               reward, next_state, done = sample
            print(h_in[0].shape, "asdsa")
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        batch = dict(
            hid_in=hidden_in,
            hid_out=hidden_out,
            act2=la_lst,
            obs=s_lst,
            obs2=ns_lst,
            act=a_lst,
            rew=r_lst,
            done=d_lst)
        return {k: torch.as_tensor(v, dtype=torch.float32)
                if type(v) != tuple else v for k, v in batch.items()}


def sac(env_fn, actor_critic=core.RNNActorCritic, ac_kwargs=dict(),
        lstm_size=256, seed=0, steps_per_epoch=4000, epochs=100,
        replay_size=int(1e3), gamma=0.99, polyak=0.995, lr=1e-3,
        alpha=.2, batch_size=2, start_steps=10000,
        update_after=1000, update_every=300, num_test_episodes=10,
        max_ep_len=200, logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # TODO: Handle discrete case
    # For discrete env, this should be measured
    # act_dim = env.action_space.n

    # Action limit for clamping: critically, assumes all dimensions share
    # the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via
    # polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    # Note, this replay buffer stores entire trajectories
    # TODO: Adjust this to also handle episodes that terminate earlier
    replay_buffer = ReplayBufferLSTM(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=lstm_size,
        size=replay_size, max_ep_len=max_ep_len)

    # Count variables (protip: try to get a feel for how different size
    # networks behave!)
    var_counts = tuple(core.count_vars(module)
                       for module in [ac.pi, ac.q1, ac.q2])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'
        % var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        print("Compute loss q")
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        hid_in, hid_out = data['hid_in'], data['hid_out']

        d = torch.unsqueeze(d, -1)
        r = torch.unsqueeze(r, -1)

        q1, _ = ac.q1(o, a, a2, hid_in)
        q2, _ = ac.q2(o, a, a2, hid_in)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, _, _, _ = ac.pi.evaluate(o2, a, hid_out)

            # Target Q-values
            # Careful, hiden are tuples (a, b)
            q1_pi_targ, _ = ac_targ.q1(o2, a2, a, hid_out)
            q2_pi_targ, _ = ac_targ.q2(o2, a2, a, hid_out)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ- alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        print("Compute loss pi")
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        # Hidden layers of the LSTM layer
        hid_in, hid_out = data['hid_in'], data['hid_out']

        pi, logp_pi, _, _, _, _ = ac.pi.evaluate(o, a2, hid_in)
        q1_pi, _ = ac.q1(o, pi, a2, hid_in)
        q2_pi, _ = ac.q2(o, pi, a2, hid_in)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        # Could apply alpha auto entropy as trade-off between
        # exploration (max entropy) and exploitation (max Q)
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(data):
        print("Updating")
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to
                # update target params, as opposed to "mul" and "add",
                # which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, a2, hidden, deterministic=False):
        # returns (action, hidden_in)
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      a2, hidden, deterministic)

    def test_agent():
        # Recurrent shape
        hidden_out = (torch.zeros([1, 1, lstm_size], dtype=torch.float),
                      torch.zeros([1, 1, lstm_size], dtype=torch.float))
        a2 = env.action_space.sample()
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                hidden_in = hidden_out
                # Take deterministic actions at test time
                a, hidden_out = get_action(o, a2, hidden_in, True)
                o, r, d, _ = test_env.step(a)
                a2 = a
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Variables for episodic replay buffer
    e_a, e_a2, e_o, e_o2, e_d, e_r = [], [], [], [], [], []

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Recurrent shape
    hidden_out = (torch.zeros([1, 1, lstm_size], dtype=torch.float),
                  torch.zeros([1, 1, lstm_size], dtype=torch.float))
    a2 = env.action_space.sample()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Set hidden_in
        hidden_in = hidden_out

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a, hidden_out = get_action(o, a2, hidden_in)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        if t == 0:
            init_hid_in = hidden_in
            init_hid_out = hidden_out

        # Episodic replay buffer
        e_a.append(a)
        e_a2.append(a2)
        e_o.append(o)
        e_o2.append(o2)
        e_d.append(d)
        e_r.append(r)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        a2 = a

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            print("End of trajectory reached")
            # Store experience to replay buffer
            e_a = np.asarray(e_a)
            e_a2 = np.asarray(e_a2)
            e_o = np.asarray(e_o)
            e_o2 = np.asarray(e_o2)
            e_d = np.asarray(e_d)
            e_r = np.asarray(e_r)

            e_a, e_a2, e_o, e_o2, e_d, e_r = [], [], [], [], [], []

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0


        # Store experience to replay buffer
        replay_buffer.store(e_a2, e_o, e_a, e_r, e_o2,
                            init_hid_in, init_hid_out, e_d)

        # Update handling
        if t >= update_after and t % update_every == 0:
            # for j in range(update_every):
            batch = replay_buffer.sample_batch(batch_size)
            update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            print("End of epoch")
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs)
