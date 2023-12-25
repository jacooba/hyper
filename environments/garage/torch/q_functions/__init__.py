"""PyTorch Q-functions."""
from environments.garage.torch.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction)
from environments.garage.torch.q_functions.discrete_cnn_q_function import (
    DiscreteCNNQFunction)
from environments.garage.torch.q_functions.discrete_dueling_cnn_q_function import (
    DiscreteDuelingCNNQFunction)
from environments.garage.torch.q_functions.discrete_mlp_q_function import (
    DiscreteMLPQFunction)

__all__ = [
    'ContinuousMLPQFunction', 'DiscreteCNNQFunction',
    'DiscreteDuelingCNNQFunction', 'DiscreteMLPQFunction'
]
