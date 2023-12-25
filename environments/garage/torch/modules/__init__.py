"""PyTorch Modules."""
# yapf: disable
# isort:skip_file
from environments.garage.torch.modules.categorical_cnn_module import CategoricalCNNModule
from environments.garage.torch.modules.cnn_module import CNNModule
from environments.garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPIndependentStdModule)  # noqa: E501
from environments.garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPTwoHeadedModule)  # noqa: E501
from environments.garage.torch.modules.gaussian_mlp_module import GaussianMLPModule
from environments.garage.torch.modules.mlp_module import MLPModule
from environments.garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
# DiscreteCNNModule must go after MLPModule
from environments.garage.torch.modules.discrete_cnn_module import DiscreteCNNModule
from environments.garage.torch.modules.discrete_dueling_cnn_module import (
    DiscreteDuelingCNNModule)
# yapf: enable

__all__ = [
    'CategoricalCNNModule',
    'CNNModule',
    'DiscreteCNNModule',
    'DiscreteDuelingCNNModule',
    'MLPModule',
    'MultiHeadedMLPModule',
    'GaussianMLPModule',
    'GaussianMLPIndependentStdModule',
    'GaussianMLPTwoHeadedModule',
]
