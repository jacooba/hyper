#!/usr/bin/env python3
"""This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Pong, and trains a DQN with 1M steps.
"""
import click
import gym

from environments.garage import wrap_experiment
from environments.garage.envs import GymEnv
from environments.garage.envs.wrappers.clip_reward import ClipReward
from environments.garage.envs.wrappers.episodic_life import EpisodicLife
from environments.garage.envs.wrappers.fire_reset import FireReset
from environments.garage.envs.wrappers.grayscale import Grayscale
from environments.garage.envs.wrappers.max_and_skip import MaxAndSkip
from environments.garage.envs.wrappers.noop import Noop
from environments.garage.envs.wrappers.resize import Resize
from environments.garage.envs.wrappers.stack_frames import StackFrames
from environments.garage.experiment.deterministic import set_seed
from environments.garage.np.exploration_policies import EpsilonGreedyPolicy
from environments.garage.replay_buffer import PathBuffer
from environments.garage.sampler import FragmentWorker, LocalSampler
from environments.garage.tf.algos import DQN
from environments.garage.tf.policies import DiscreteQFArgmaxPolicy
from environments.garage.tf.q_functions import DiscreteCNNQFunction
from environments.garage.trainer import TFTrainer


@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
@click.option('--max_episode_length', type=int, default=500)
@wrap_experiment
def dqn_pong(ctxt=None, seed=1, buffer_size=int(5e4), max_episode_length=500):
    """Train DQN on PongNoFrameskip-v4 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        buffer_size (int): Number of timesteps to store in replay buffer.
        max_episode_length (int): Maximum length of an episode, after which an
            episode is considered complete. This is used during testing to
            minimize the memory required to store a single episode.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        n_epochs = 100
        steps_per_epoch = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = gym.make('PongNoFrameskip-v4')
        env = env.unwrapped
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = GymEnv(env, is_image=True, max_episode_length=max_episode_length)

        replay_buffer = PathBuffer(capacity_in_transitions=buffer_size)

        qf = DiscreteCNNQFunction(env_spec=env.spec,
                                  filters=(
                                              (32, (8, 8)),
                                              (64, (4, 4)),
                                              (64, (3, 3)),
                                          ),
                                  strides=(4, 2, 1),
                                  dueling=False)  # yapf: disable

        policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.02,
                                                 decay_ratio=0.1)

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=FragmentWorker)

        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_policy=exploration_policy,
                   replay_buffer=replay_buffer,
                   sampler=sampler,
                   qf_lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   double_q=False,
                   n_train_steps=500,
                   steps_per_epoch=steps_per_epoch,
                   target_network_update_freq=2,
                   buffer_batch_size=32)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_pong()
