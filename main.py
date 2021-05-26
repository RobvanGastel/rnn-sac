# autopep8: off
import gym
import torch

import sys
sys.path.append('rnn-sac')

from rnn_sac import sac
import core as core
# autopep8: on


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--exp_name', type=str, default='sac')
    # args = parser.parse_args()

    # Two possibilities, ML1/ML10 and the current envs from HER.
    # names = ['FetchReach-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1',
    #          'FetchPush-v1']

    max_ep_len = gym.make('Pendulum-v0')._max_episode_steps
    batch_size = 5

    args = {'env': gym.make('Pendulum-v0'), 'hid': 256, 'lr': 3e-3,
            'alpha': 0.2,  'l': 2, 'gamma': 0.99, 'seed': 0, 'epoch': 30,
            'batch_size': batch_size,
            'start_steps': 10000,
            'update_after': max_ep_len*batch_size,
            'update_every': 50,
            'num_test_episodes': 10,
            'max_ep_len': max_ep_len,
            'exp_name': 'SAC', 'auto_entropy': True}

    args['exp_name'] += f"_{'pendulum'}_{args['alpha']}_{args['lr']}"

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args['exp_name'], args['seed'])

    torch.set_num_threads(torch.get_num_threads())

    # TODO: Tune parameters, parameter grid?
    sac(lambda: args['env'], actor_critic=core.GRUActorCritic,
        ac_kwargs=dict(hidden_size=args['hid']),
        hidden_size=args['hid'], seed=args['seed'], lr=args['lr'],
        alpha=args['alpha'], gamma=args['gamma'], epochs=args['epoch'],
        batch_size=args['batch_size'], start_steps=args['start_steps'],
        update_after=args['update_after'], max_ep_len=args['max_ep_len'],
        update_every=args['update_every'],
        num_test_episodes=args['num_test_episodes'],
        logger_kwargs=logger_kwargs, auto_entropy=args['auto_entropy'])
