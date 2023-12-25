"""Tensorflow implementation of reinforcement learning algorithms."""
from environments.garage.tf.algos.ddpg import DDPG
from environments.garage.tf.algos.dqn import DQN
from environments.garage.tf.algos.erwr import ERWR
from environments.garage.tf.algos.npo import NPO
from environments.garage.tf.algos.ppo import PPO
from environments.garage.tf.algos.reps import REPS
from environments.garage.tf.algos.rl2 import RL2
from environments.garage.tf.algos.rl2ppo import RL2PPO
from environments.garage.tf.algos.rl2trpo import RL2TRPO
from environments.garage.tf.algos.td3 import TD3
from environments.garage.tf.algos.te_npo import TENPO
from environments.garage.tf.algos.te_ppo import TEPPO
from environments.garage.tf.algos.tnpg import TNPG
from environments.garage.tf.algos.trpo import TRPO
from environments.garage.tf.algos.vpg import VPG

__all__ = [
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'PPO',
    'REPS',
    'RL2',
    'RL2PPO',
    'RL2TRPO',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
    'TENPO',
    'TEPPO',
]
