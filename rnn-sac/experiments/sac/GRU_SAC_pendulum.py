# autopep8: off
import gym
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import gru_sac as algo
import core as core
# autopep8: on


if __name__ == "__main__":
    max_ep_len = gym.make('Pendulum-v0')._max_episode_steps

    hids = [512]  # , 1024]
    lrs = [3e-3]
    batch_sizes = [5]
    entrop = [True]

    for ent in entrop:
        for lr in lrs:
            for batch_size in batch_sizes:
                args = {'env': 'Pendulum-v0', 'hid': 128, 'lr': lr,
                        'alpha': 0.2,
                        'l': 2, 'gamma': 0.99, 'seed': 0, 'epoch': 30,
                        'batch_size': batch_size,
                        'start_steps': 10000,
                        'update_after': max_ep_len*batch_size,
                        'update_every': 50,
                        'num_test_episodes': 10, 'max_ep_len': max_ep_len,
                        'exp_name': 'GRU_SAC', 'auto_entropy': ent}

                # args['exp_name'] += f"_Pendulum_alpha{args['alpha']}_lr{args['lr']}_batch{batch_size}_hidden{hid}"
                args['exp_name'] += f"_Pendulum"

                from algo.utils.run_utils import setup_logger_kwargs
                logger_kwargs = setup_logger_kwargs(
                    args['exp_name'], args['seed'])

                # TODO: Add tensorboard logger
                # Customer logger for tensorboard
                # writer = SummaryWriter('./meta-learners/algo/data/test')

                torch.set_num_threads(torch.get_num_threads())

                # TODO: Tune parameters, parameter grid?
                algo.sac(lambda: gym.make(args['env']),
                         actor_critic=core.GRUActorCritic,
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
                         logger_kwargs=logger_kwargs,
                         auto_entropy=args['auto_entropy'])
