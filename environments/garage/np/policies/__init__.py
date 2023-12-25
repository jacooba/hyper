"""Policies which use NumPy as a numerical backend."""

from environments.garage.np.policies.fixed_policy import FixedPolicy
from environments.garage.np.policies.policy import Policy
from environments.garage.np.policies.scripted_policy import ScriptedPolicy
from environments.garage.np.policies.uniform_random_policy import UniformRandomPolicy

__all__ = ['FixedPolicy', 'Policy', 'ScriptedPolicy', 'UniformRandomPolicy']
