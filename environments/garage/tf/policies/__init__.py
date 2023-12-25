"""Policies for TensorFlow-based algorithms."""
from environments.garage.tf.policies.categorical_cnn_policy import CategoricalCNNPolicy
from environments.garage.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from environments.garage.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from environments.garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from environments.garage.tf.policies.continuous_mlp_policy import ContinuousMLPPolicy
from environments.garage.tf.policies.discrete_qf_argmax_policy import DiscreteQFArgmaxPolicy
from environments.garage.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from environments.garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from environments.garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from environments.garage.tf.policies.gaussian_mlp_task_embedding_policy import (
    GaussianMLPTaskEmbeddingPolicy)
from environments.garage.tf.policies.policy import Policy
from environments.garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy

__all__ = [
    'Policy', 'CategoricalCNNPolicy', 'CategoricalGRUPolicy',
    'CategoricalLSTMPolicy', 'CategoricalMLPPolicy', 'ContinuousMLPPolicy',
    'DiscreteQFArgmaxPolicy', 'GaussianGRUPolicy', 'GaussianLSTMPolicy',
    'GaussianMLPPolicy', 'GaussianMLPTaskEmbeddingPolicy',
    'TaskEmbeddingPolicy'
]
