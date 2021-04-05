# import gym
# import torch

# # from algo import sac as algo
# # import algo.core
# import algo.rnn_sac as rnn_algo
# import algo.sac as algo
# import algo.core as core

# # For testing purposes,
# if __name__ == "__main__":
#     # TODO: Test on BipedalWalker-v3
#     # TODO: meta-world ML1 and ML10
#     args = {'env': 'Pendulum-v0', 'hid': 256,
#             'l': 2, 'gamma': 0.99, 'seed': 0, 'epoch': 50,
#             'exp_name': 'rnn_sac', 'seed': 42}

#     from algo.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args['exp_name'], args['seed'])

#     torch.set_num_threads(torch.get_num_threads())

#     # RNN SAC
#     rnn_algo.sac(lambda: gym.make(args['env']),
#                  actor_critic=core.RNNActorCritic, ac_kwargs=dict(
#                      hidden_size=args['hid']),
#                  lstm_size=args['hid'], logger_kwargs=logger_kwargs)

#     # SAC
#     # algo.sac(lambda: gym.make(args['env']), actor_critic=core.MLPActorCritic,
#     #             ac_kwargs=dict(hidden_sizes=[args['hid']]*args['l']),
#     #             logger_kwargs=logger_kwargs)
