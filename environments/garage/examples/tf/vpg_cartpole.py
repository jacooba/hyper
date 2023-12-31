#!/usr/bin/env python3
"""This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 16
"""
from environments.garage import wrap_experiment
from environments.garage.envs import GymEnv
from environments.garage.experiment.deterministic import set_seed
from environments.garage.np.baselines import LinearFeatureBaseline
from environments.garage.sampler import RaySampler
from environments.garage.tf.algos import VPG
from environments.garage.tf.policies import CategoricalMLPPolicy
from environments.garage.trainer import TFTrainer


@wrap_experiment
def vpg_cartpole(ctxt=None, seed=1):
    """Train VPG with CartPole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=10000)


vpg_cartpole()
