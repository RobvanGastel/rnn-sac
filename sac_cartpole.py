import gym
import torch

# from algo import sac as algo
# import algo.core
import algo.sac as algo
import algo.core as core

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')

    args = {'env': 'HalfCheetah-v2', 'hid': 256,
            'l': 2, 'gamma': 0.99, 'seed': 0, 'epoch': 50,
            'exp_name': 'sac'}

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    algo.sac(lambda: gym.make(args['env']), actor_critic=core.RNNActorCritic,
             ac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']))
