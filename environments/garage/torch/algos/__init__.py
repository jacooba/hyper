"""PyTorch algorithms."""
# isort:skip_file

from environments.garage.torch.algos.bc import BC
from environments.garage.torch.algos.ddpg import DDPG
# VPG has to be imported first because it is depended by PPO and TRPO.
# PPO, TRPO, and VPG need to be imported before their MAML variants
from environments.garage.torch.algos.dqn import DQN
from environments.garage.torch.algos.vpg import VPG
from environments.garage.torch.algos.maml_vpg import MAMLVPG
from environments.garage.torch.algos.ppo import PPO
from environments.garage.torch.algos.maml_ppo import MAMLPPO
from environments.garage.torch.algos.td3 import TD3
from environments.garage.torch.algos.trpo import TRPO
from environments.garage.torch.algos.maml_trpo import MAMLTRPO
# SAC needs to be imported before MTSAC
from environments.garage.torch.algos.sac import SAC
from environments.garage.torch.algos.mtsac import MTSAC
from environments.garage.torch.algos.pearl import PEARL

__all__ = [
    'BC', 'DDPG', 'DQN', 'VPG', 'PPO', 'TD3', 'TRPO', 'MAMLPPO', 'MAMLTRPO',
    'MAMLVPG', 'MTSAC', 'PEARL', 'SAC'
]
