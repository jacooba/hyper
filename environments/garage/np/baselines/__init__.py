"""Baselines (value functions) which use NumPy as a numerical backend."""
from environments.garage.np.baselines.baseline import Baseline
from environments.garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from environments.garage.np.baselines.linear_multi_feature_baseline import (
    LinearMultiFeatureBaseline)
from environments.garage.np.baselines.zero_baseline import ZeroBaseline

__all__ = [
    'Baseline', 'LinearFeatureBaseline', 'LinearMultiFeatureBaseline',
    'ZeroBaseline'
]
