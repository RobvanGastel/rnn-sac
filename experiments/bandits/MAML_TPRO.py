import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage import TFTrainer
from garage.np.baselines import LinearFeatureBaseline
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.algos import TRPO

@wrap_experiment
def maml_tpro_rnn_bandits_push(ctxt, seed=42, n_epochs=1, batch_size_per_task=20):
    set_seed(seed)
    envs = [GymEnv('CartPole-v0'), GymEnv('CartPole-v0'), GymEnv('CartPole-v0')]
    env = MultiEnvWrapper(envs)
    
    latent_length = 2
    inference_window = 6
    batch_size = batch_size_per_task
    policy_ent_coeff = 2e-2
    encoder_ent_coeff = 2e-4
    inference_ce_coeff = 5e-2
    max_episode_length = 100
    embedding_init_std = 0.1
    embedding_max_std = 0.2
    embedding_min_std = 1e-6
    policy_init_std = 1.0
    policy_max_std = None
    policy_min_std = None

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))


        sampler = LocalSampler(agents=policy,
                               envs=env,
                              max_episode_length=200)

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env_spec=env.spec,
            sampler=sampler,
            policy=policy,
            baseline=baseline,
            discount=0.99,
            max_kl_step=0.01,
        )
        

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=False)


maml_tpro_rnn_bandits_push()