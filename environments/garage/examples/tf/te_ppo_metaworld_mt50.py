#!/usr/bin/env python3
"""This is an example to train Task Embedding PPO with PointEnv."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import tensorflow as tf

from environments.garage import wrap_experiment
from environments.garage.envs import normalize
from environments.garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from environments.garage.experiment.deterministic import set_seed
from environments.garage.experiment.task_sampler import MetaWorldTaskSampler
from environments.garage.np.baselines import LinearMultiFeatureBaseline
from environments.garage.sampler import LocalSampler
from environments.garage.tf.algos import TEPPO
from environments.garage.tf.algos.te import TaskEmbeddingWorker
from environments.garage.tf.embeddings import GaussianMLPEncoder
from environments.garage.tf.policies import GaussianMLPTaskEmbeddingPolicy
from environments.garage.trainer import TFTrainer


@click.command()
@click.option('--seed', default=1)
@click.option('--n_epochs', default=600)
@click.option('--batch_size_per_task', default=1024)
@click.option('--n_tasks', default=50)
@wrap_experiment
def te_ppo_mt50(ctxt, seed, n_epochs, batch_size_per_task, n_tasks):
    """Train Task Embedding PPO with PointEnv.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_epochs (int): Total number of epochs for training.
        batch_size_per_task (int): Batch size of samples for each task.
        n_tasks (int): Number of tasks to use. Should be a multiple of 50.

    """
    set_seed(seed)
    mt50 = metaworld.MT50()
    task_sampler = MetaWorldTaskSampler(mt50,
                                        'train',
                                        lambda env, _: normalize(env),
                                        add_env_onehot=False)
    assert n_tasks % 50 == 0
    assert n_tasks <= 2500
    envs = [env_up() for env_up in task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    latent_length = 6
    inference_window = 6
    batch_size = batch_size_per_task * n_tasks
    policy_ent_coeff = 2e-2
    encoder_ent_coeff = 2e-4
    inference_ce_coeff = 5e-2
    embedding_init_std = 0.1
    embedding_max_std = 0.2
    embedding_min_std = 1e-6
    policy_init_std = 1.0
    policy_max_std = None
    policy_min_std = None

    with TFTrainer(snapshot_config=ctxt) as trainer:

        task_embed_spec = TEPPO.get_encoder_spec(env.task_space,
                                                 latent_dim=latent_length)

        task_encoder = GaussianMLPEncoder(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=embedding_init_std,
            max_std=embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        traj_embed_spec = TEPPO.get_infer_spec(
            env.spec,
            latent_dim=latent_length,
            inference_window_size=inference_window)

        inference = GaussianMLPEncoder(
            name='inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),
            std_share_network=True,
            init_std=2.0,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        policy = GaussianMLPTaskEmbeddingPolicy(
            name='policy',
            env_spec=env.spec,
            encoder=task_encoder,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=policy_max_std,
            init_std=policy_init_std,
            min_std=policy_min_std,
        )

        baseline = LinearMultiFeatureBaseline(
            env_spec=env.spec, features=['observations', 'tasks', 'latents'])

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=TaskEmbeddingWorker)

        algo = TEPPO(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     sampler=sampler,
                     inference=inference,
                     discount=0.99,
                     lr_clip_range=0.2,
                     policy_ent_coeff=policy_ent_coeff,
                     encoder_ent_coeff=encoder_ent_coeff,
                     inference_ce_coeff=inference_ce_coeff,
                     use_softplus_entropy=True,
                     optimizer_args=dict(
                         batch_size=32,
                         max_optimization_epochs=10,
                         learning_rate=1e-3,
                     ),
                     inference_optimizer_args=dict(
                         batch_size=32,
                         max_optimization_epochs=10,
                     ),
                     center_adv=True,
                     stop_ce_gradient=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=False)


te_ppo_mt50()
