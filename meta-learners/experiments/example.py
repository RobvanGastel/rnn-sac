#!/usr/bin/env python3
"""This is an example to train MAML-VPG on HalfCheetahDirEnv environment."""
# pylint: disable=no-value-for-parameter
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.algos import TD3
from garage.sampler import FragmentWorker, LocalSampler
from garage.replay_buffer import PathBuffer
from garage.np.policies import UniformRandomPolicy
from garage.np.exploration_policies import AddGaussianNoise
from torch.nn import functional as F
import click
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.mujoco import HalfCheetahDirEnv
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=300)
@click.option('--episodes_per_task', default=40)
@click.option('--meta_batch_size', default=20)
@wrap_experiment(snapshot_mode='all')
def maml_ppo_half_cheetah_dir(ctxt, seed, epochs, episodes_per_task,
                              meta_batch_size):
    """Set up environment and algorithm and run the task.
    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.
    """
    set_seed(seed)
    max_episode_length = 100
    env = normalize(GymEnv(HalfCheetahDirEnv(),
                           max_episode_length=max_episode_length),
                    expected_action_scale=10.)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    task_sampler = SetTaskSampler(
        HalfCheetahDirEnv,
        wrapper=lambda env, _: normalize(GymEnv(
            env, max_episode_length=max_episode_length),
            expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=2,
                                   n_test_episodes=10)

    trainer = Trainer(ctxt)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    algo = MAMLPPO(env=env,
                   policy=policy,
                   sampler=sampler,
                   task_sampler=task_sampler,
                   value_function=value_function,
                   meta_batch_size=meta_batch_size,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * env.spec.max_episode_length)


@wrap_experiment(snapshot_mode='none')
def td3_half_cheetah(ctxt=None, seed=1):
    """Train TD3 with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
        determinism.
    """
    set_seed(seed)

    n_epochs = 500
    steps_per_epoch = 20
    sampler_batch_size = 250
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

    trainer = Trainer(ctxt)
    env = normalize(GymEnv('HalfCheetah-v2'))

    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    exploration_policy = AddGaussianNoise(env.spec,
                                          policy,
                                          total_timesteps=num_timesteps,
                                          max_sigma=0.1,
                                          min_sigma=0.1)

    uniform_random_policy = UniformRandomPolicy(env.spec)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=exploration_policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    td3 = TD3(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              replay_buffer=replay_buffer,
              sampler=sampler,
              policy_optimizer=torch.optim.Adam,
              qf_optimizer=torch.optim.Adam,
              exploration_policy=exploration_policy,
              uniform_random_policy=uniform_random_policy,
              target_update_tau=0.005,
              discount=0.99,
              policy_noise_clip=0.5,
              policy_noise=0.2,
              policy_lr=1e-3,
              qf_lr=1e-3,
              steps_per_epoch=40,
              start_steps=1000,
              grad_steps_per_env_step=50,
              min_buffer_size=1000,
              buffer_batch_size=100)

    trainer.setup(algo=td3, env=env)
    trainer.train(n_epochs=750, batch_size=100)


td3_half_cheetah(seed=0)


# maml_ppo_half_cheetah_dir()
