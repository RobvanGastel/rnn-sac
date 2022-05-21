import gym
import argparse

from rnn_sac.sac.sac import SAC
from rnn_sac.utils.run_utils import setup_logger_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='CartPole-v1')

    parser.add_argument('--env', type=str, default='Pendulum-v0')

    parser.add_argument('--use_meta_learning', action='store_true')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument("--exploration_sampling", action="store_true")
    parser.add_argument('--auto_entropy', action='store_true')
    parser.add_argument("--test_trial", type=int,
                        default=20, help="execute test trial every n")
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    env = gym.make(args.env)

    ac_kwargs = dict(hidden_size=[args.hid]*2)
    agent = SAC(env,
                logger_kwargs=logger_kwargs,
                seed=args.seed,
                gamma=args.gamma,
                ac_kwargs=ac_kwargs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                update_every=args.update_every,
                hidden_size=args.hid,
                exploration_sampling=args.exploration_sampling)

    agent.train_agent(env)
