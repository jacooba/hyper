"""Q-Functions for TensorFlow-based algorithms."""
# isort:skip_file

from environments.garage.tf.q_functions.continuous_cnn_q_function import (
    ContinuousCNNQFunction)
from environments.garage.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction)
from environments.garage.tf.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from environments.garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from environments.garage.tf.q_functions.discrete_mlp_dueling_q_function import (
    DiscreteMLPDuelingQFunction)

__all__ = [
    'ContinuousMLPQFunction', 'DiscreteCNNQFunction', 'DiscreteMLPQFunction',
    'DiscreteMLPDuelingQFunction', 'ContinuousCNNQFunction'
]
