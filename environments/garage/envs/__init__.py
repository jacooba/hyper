"""Garage wrappers for gym environments."""

from environments.garage.envs.grid_world_env import GridWorldEnv
from environments.garage.envs.gym_env import GymEnv
from environments.garage.envs.metaworld_set_task_env import MetaWorldSetTaskEnv
from environments.garage.envs.multi_env_wrapper import MultiEnvWrapper
from environments.garage.envs.normalized_env import normalize
from environments.garage.envs.point_env import PointEnv
from environments.garage.envs.task_name_wrapper import TaskNameWrapper
from environments.garage.envs.task_onehot_wrapper import TaskOnehotWrapper

__all__ = [
    'GymEnv',
    'GridWorldEnv',
    'MetaWorldSetTaskEnv',
    'MultiEnvWrapper',
    'normalize',
    'PointEnv',
    'TaskOnehotWrapper',
    'TaskNameWrapper',
]
