"""Samplers which run agents that use Tensorflow in environments."""

from environments.garage.tf.samplers.worker import TFWorkerClassWrapper, TFWorkerWrapper

__all__ = ['TFWorkerClassWrapper', 'TFWorkerWrapper']
