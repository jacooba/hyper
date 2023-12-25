#!/usr/bin/env python3
"""This is an example to train MAML-VPG on HalfCheetahDirEnv environment."""
# pylint: disable=no-value-for-parameter
import click
import torch

from environments.garage import wrap_experiment
from environments.garage.envs import GymEnv, normalize
from environments.garage.envs.mujoco import HalfCheetahDirEnv
from environments.garage.experiment import MetaEvaluator
from environments.garage.experiment.deterministic import set_seed
from environments.garage.experiment.task_sampler import SetTaskSampler
from environments.garage.sampler import RaySampler
from environments.garage.torch.algos import MAMLVPG
from environments.garage.torch.policies import GaussianMLPPolicy
from environments.garage.torch.value_functions import GaussianMLPValueFunction
from environments.garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=300)
@click.option('--episodes_per_task', default=40)
@click.option('--meta_batch_size', default=20)
@wrap_experiment(snapshot_mode='all')
def maml_vpg_half_cheetah_dir(ctxt, seed, epochs, episodes_per_task,
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
    env = normalize(GymEnv(HalfCheetahDirEnv(), max_episode_length=100),
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

    max_episode_length = env.spec.max_episode_length

    task_sampler = SetTaskSampler(
        HalfCheetahDirEnv,
        wrapper=lambda env, _: normalize(GymEnv(
            env, max_episode_length=max_episode_length),
                                         expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=1,
                                   n_test_episodes=10)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)

    trainer = Trainer(ctxt)
    algo = MAMLVPG(env=env,
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
                  batch_size=episodes_per_task * max_episode_length)


maml_vpg_half_cheetah_dir()
