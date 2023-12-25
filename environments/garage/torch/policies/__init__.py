"""PyTorch Policies."""
from environments.garage.torch.policies.categorical_cnn_policy import CategoricalCNNPolicy
from environments.garage.torch.policies.context_conditioned_policy import (
    ContextConditionedPolicy)
from environments.garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from environments.garage.torch.policies.discrete_cnn_policy import DiscreteCNNPolicy
from environments.garage.torch.policies.discrete_qf_argmax_policy import (
    DiscreteQFArgmaxPolicy)
from environments.garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from environments.garage.torch.policies.policy import Policy
from environments.garage.torch.policies.tanh_gaussian_mlp_policy import (
    TanhGaussianMLPPolicy)

__all__ = [
    'CategoricalCNNPolicy',
    'DeterministicMLPPolicy',
    'DiscreteCNNPolicy',
    'DiscreteQFArgmaxPolicy',
    'GaussianMLPPolicy',
    'Policy',
    'TanhGaussianMLPPolicy',
    'ContextConditionedPolicy',
]
