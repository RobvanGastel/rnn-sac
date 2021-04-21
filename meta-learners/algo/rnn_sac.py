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


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


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

    def sample_batch(self, batch_size=32):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, \
            ci_lst, ho_lst, co_lst, d_lst = [
            ], [], [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        # TODO: Omit this for-loop by moving it to torch/np
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, \
                reward, next_state, done = sample
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
        # cat along the batch dim
        hi_lst = torch.cat(hi_lst, dim=-2).detach()
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

        return {k: torch.FloatTensor(v).cuda()
                if type(v) != tuple else v for k, v in batch.items()}


def sac(env_fn, actor_critic=core.RNNActorCritic, ac_kwargs=dict(),
        hidden_size=512, seed=0, steps_per_epoch=4000, epochs=100,
        replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=3e-4,
        alpha=0., batch_size=1, start_steps=10000,
        update_after=1000, update_every=200, num_test_episodes=10,
        max_ep_len=200, logger_kwargs=dict(), save_freq=1,
        rl2=False):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an
            ``act`` method, a ``pi`` module, a ``q1`` module, and a ``q2``
            module. The ``act`` method and ``pi`` module should accept batches
            of observations as inputs, and ``q1`` and ``q2`` should accept
            a batch of observations and a batch of actions as inputs. When
            called, ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ====================================
            Call         Output Shape      Description
            ===========  ================  ====================================
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
            ===========  ================  ====================================

            Calling ``pi`` should return:

            ===========  ================  ====================================
            Symbol       Shape             Description
            ===========  ================  ====================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ====================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action
            pairs) for the agent and the environment in each epoch.

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
    # TODO: If alpha SGD is enabled for exploration
    # target_entropy = -2
    # reward_scale = 10.
    # auto_entropy = True

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = NormalizedActions(env_fn()), NormalizedActions(env_fn())
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Take gradient of alpha to balance exploitation vs exploration
    # TODO: How does this work in meta-learning setting?
    # Garage proposes different alphas for every task,
    # https://garage.readthedocs.io/en/latest/user/algo_mtsac.html
    # log_alpha = torch.zeros(
    #     1, dtype=torch.float32, requires_grad=True)

    # Freeze target networks with respect to optimizers (only update via
    # polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    # Note, this replay buffer stores entire trajectories
    replay_buffer = ReplayBufferLSTM(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_size,
        size=replay_size, max_ep_len=max_ep_len)

    # Count variables (protip: try to get a feel for how different size
    # networks behave!)
    var_counts = tuple(core.count_vars(module)
                       for module in [ac.pi, ac.q1, ac.q2])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'
        % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        # Hidden layers of the LSTM layer
        hid_in, hid_out = data['hid_in'], data['hid_out']

        d = torch.unsqueeze(d, -1)
        r = torch.unsqueeze(r, -1)

        # TODO: Normalize the batch reward
        # normalize with batch mean and std; plus a small number to prevent
        # numerical problem
        # r = reward_scale * \
        #     (r - r.mean(dim=0)) / (r.std(dim=0) + 1e-6)

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
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        # Hidden layers of the LSTM layer
        hid_in, hid_out = data['hid_in'], data['hid_out']

        pi, logp_pi, _, _, _, _ = ac.pi.evaluate(o, a2, hid_in)
        q1_pi, _ = ac.q1(o, pi, a2, hid_in)
        q2_pi, _ = ac.q2(o, pi, a2, hid_in)
        q_pi = torch.min(q1_pi, q2_pi)

        # TODO: Possibility of adding decaying alpha
        # Could apply alpha auto entropy as trade-off between
        # exploration (max entropy) and exploitation (max Q)
        # if auto_entropy is True:
        #     alpha_loss = -(log_alpha * (logp_pi +
        #                                 target_entropy).detach()).mean()
        #     alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     alpha_optimizer.step()
        #     alpha = log_alpha.exp()

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    # TODO: If exploring alpha loss
    # alpha_optimizer = Adam([log_alpha], lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # print("Updating")

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
        hidden_out = (torch.zeros([1, 1, hidden_size],
                                  dtype=torch.float).cuda(),
                      torch.zeros([1, 1, hidden_size],
                                  dtype=torch.float).cuda())
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
    hidden_out = (torch.zeros([1, 1, hidden_size], dtype=torch.float).cuda(),
                  torch.zeros([1, 1, hidden_size], dtype=torch.float).cuda())
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
            # print("End of trajectory reached")
            # Store experience to replay buffer
            e_a = np.asarray(e_a)
            e_a2 = np.asarray(e_a2)
            e_o = np.asarray(e_o)
            e_o2 = np.asarray(e_o2)
            e_d = np.asarray(e_d)
            e_r = np.asarray(e_r)
            replay_buffer.store(e_a2, e_o, e_a, e_r, e_o2,
                                init_hid_in, init_hid_out, e_d)
            e_a, e_a2, e_o, e_o2, e_d, e_r = [], [], [], [], [], []

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            # for j in range(update_every):
            batch = replay_buffer.sample_batch(batch_size)
            update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # print("End of epoch")
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # TODO: Add tensorboard logger
            # if writer is not None:
            #     writer.add_scalar('Average', ep_ret, epoch)
            #     writer.flush()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    # if writer is not None:
    #     writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from algo.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    # TODO: Solve for RNN SAC
    # sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
    #     gamma=args.gamma, seed=args.seed, epochs=args.epochs)
