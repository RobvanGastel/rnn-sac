# autopep8: off
import gym
import torch

import sys
import argparse
sys.path.append('rnn-sac')

from utils.run_utils import setup_logger_kwargs

from rnn_sac import sac as rnn_sac
from rl2_sac import sac as rl2_sac
from sac import sac

import core as core
# autopep8: on


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--rnn_cell', type=str, default='GRU')
    parser.add_argument('--meta_learning', action='store_true')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--auto_entropy', action='store_true')
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    max_ep_len = gym.make('Pendulum-v0')._max_episode_steps
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    torch.set_num_threads(torch.get_num_threads())

    if args.meta_learning:
        rl2_sac(args.env,
                actor_critic=core.GRUActorCritic if args.rnn_cell == 'GRU' else core.LSTMActorCritic,
                rnn_cell=args.rnn_cell,
                ac_kwargs=dict(hidden_size=args.hid),
                hidden_size=args.hid,
                seed=args.seed,
                lr=args.lr,
                alpha=args.alpha,
                gamma=args.gamma,
                epochs=args.epochs,
                batch_size=args.batch_size,
                start_steps=10000,
                update_after=max_ep_len*args.batch_size,
                max_ep_len=max_ep_len,
                update_every=args.update_every,
                num_test_episodes=args.num_test_episodes,
                logger_kwargs=logger_kwargs,
                env_wrapper=None,  # TODO: Fix this env_wrapper
                auto_entropy=args.auto_entropy)
    else:
        if args.rnn_cell != 'MLP':
            rnn_sac(lambda: gym.make(args.env),
                    actor_critic=core.GRUActorCritic if args.rnn_cell == 'GRU' else core.LSTMActorCritic,
                    rnn_cell=args.rnn_cell,
                    ac_kwargs=dict(hidden_size=args.hid),
                    hidden_size=args.hid,
                    seed=args.seed,
                    lr=args.lr,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    start_steps=10000,
                    update_after=max_ep_len*args.batch_size,
                    max_ep_len=max_ep_len,
                    update_every=args.update_every,
                    num_test_episodes=args.num_test_episodes,
                    logger_kwargs=logger_kwargs,
                    env_wrapper=None,  # TODO: Fix this env_wrapper
                    auto_entropy=args.auto_entropy)
        else:
            # Default SAC implementation to compare baselines
            sac(lambda: gym.make(args.env),
                actor_critic=core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*2),
                seed=args.seed,
                lr=args.lr,
                alpha=args.alpha,
                gamma=args.gamma,
                epochs=args.epochs,
                batch_size=args.batch_size,
                start_steps=10000,
                update_after=max_ep_len,
                max_ep_len=max_ep_len,
                update_every=args.update_every,
                num_test_episodes=args.num_test_episodes,
                logger_kwargs=logger_kwargs)
