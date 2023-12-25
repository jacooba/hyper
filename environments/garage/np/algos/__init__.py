"""Reinforcement learning algorithms which use NumPy as a numerical backend."""
from environments.garage.np.algos.cem import CEM
from environments.garage.np.algos.cma_es import CMAES
from environments.garage.np.algos.meta_rl_algorithm import MetaRLAlgorithm
from environments.garage.np.algos.nop import NOP
from environments.garage.np.algos.rl_algorithm import RLAlgorithm

__all__ = [
    'RLAlgorithm',
    'CEM',
    'CMAES',
    'MetaRLAlgorithm',
    'NOP',
]
