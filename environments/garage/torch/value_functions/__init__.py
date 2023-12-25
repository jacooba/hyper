"""Value functions which use PyTorch."""
from environments.garage.torch.value_functions.gaussian_mlp_value_function import (
    GaussianMLPValueFunction)
from environments.garage.torch.value_functions.value_function import ValueFunction

__all__ = ['ValueFunction', 'GaussianMLPValueFunction']
