# autopep8: off
import gym
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import algo.rnn_sac as algo
import algo.core as core
# autopep8: on


if __name__ == "__main__":
    max_ep_len = gym.make('Pendulum-v0')._max_episode_steps

    alpha = np.geomspace(0.7, 0.1, num=4)
    hids = [2**7, 2**8, 2**9]
    lrs = [0.1, 0.05, 0.01, 0.005]
    batch_sizes = [3, 5, 10, 20]

    for hid in hids:
        for lr in lrs:
            for batch_size in batch_sizes:
                args = {'env': 'Pendulum-v0', 'hid': hid, 'lr': lr,
                        'alpha': 0.2,
                        'l': 2, 'gamma': 0.99, 'seed': 0, 'epoch': 50,
                        'batch_size': batch_size,
                        'start_steps': 10000,
                        'update_after': max_ep_len*batch_size,
                        'update_every': 50,
                        'num_test_episodes': 10, 'max_ep_len': max_ep_len,
                        'exp_name': 'LSTM_SAC'}

                args['exp_name'] += f"_Pendulum_{args['alpha']}_{args['lr']} \
                    _{batch_size}_{hid}"

                from algo.utils.run_utils import setup_logger_kwargs
                logger_kwargs = setup_logger_kwargs(
                    args['exp_name'], args['seed'])

                # TODO: Add tensorboard logger
                # Customer logger for tensorboard
                # writer = SummaryWriter('./meta-learners/algo/data/test')

                torch.set_num_threads(torch.get_num_threads())

                # TODO: Tune parameters, parameter grid?
                algo.sac(lambda: gym.make(args['env']),
                         actor_critic=core.RNNActorCritic,
                         ac_kwargs=dict(hidden_size=args['hid']),
                         hidden_size=args['hid'],
                         seed=args['seed'], lr=args['lr'],
                         alpha=args['alpha'],
                         gamma=args['gamma'], epochs=args['epoch'],
                         batch_size=args['batch_size'],
                         start_steps=args['start_steps'],
                         update_after=args['update_after'],
                         max_ep_len=args['max_ep_len'],
                         update_every=args['update_every'],
                         num_test_episodes=args['num_test_episodes'],
                         logger_kwargs=logger_kwargs)
