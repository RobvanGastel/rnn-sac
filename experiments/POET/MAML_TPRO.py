import tensorflow as tf
import torch

from garage.experiment.deterministic import set_seed
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper

from garage import TFTrainer

from garage.np.baselines import LinearFeatureBaseline
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.tf.policies import GaussianMLPPolicy

from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.experiment import MetaEvaluator

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.mujoco import HalfCheetahDirEnv
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@wrap_experiment
def meta_maml_tpro_rnn_bipedal(ctxt, seed=42, n_epochs=100,
                               batch_size_per_task=20, max_episode_length=200):

    # Set hyperparameters
    batch_size = batch_size_per_task
    meta_batch_size = 20
    episodes_per_task = 40
    epochs = 300
    set_seed(seed)

    # MDPs
    tasks = [
        GymEnv('BipedalWalker-v3'),
        GymEnv('BipedalWalkerHardcore-v3')
    ]
    env = MultiEnvWrapper(tasks)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=[32, 32],
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    trainer = Trainer(ctxt)

    task_sampler = SetTaskSampler(env,
                                  env=tasks[0])
    # task_sampler = SetTaskSampler(
    #     HalfCheetahDirEnv,
    # wrapper=lambda env, _: normalize(GymEnv(
    #     env, max_episode_length=max_episode_length),
    #                                  expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=2,
                                   n_test_episodes=10)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=max_episode_length)

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


meta_maml_tpro_rnn_bipedal()
