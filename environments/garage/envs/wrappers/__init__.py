"""gym.Env wrappers.

Used to transform an environment in a modular way.
It is also possible to apply multiple wrappers at the same
time.

Example:
    StackFrames(GrayScale(gym.make('env')))

"""
from environments.garage.envs.wrappers.atari_env import AtariEnv
from environments.garage.envs.wrappers.clip_reward import ClipReward
from environments.garage.envs.wrappers.episodic_life import EpisodicLife
from environments.garage.envs.wrappers.fire_reset import FireReset
from environments.garage.envs.wrappers.grayscale import Grayscale
from environments.garage.envs.wrappers.max_and_skip import MaxAndSkip
from environments.garage.envs.wrappers.noop import Noop
from environments.garage.envs.wrappers.pixel_observation import PixelObservationWrapper
from environments.garage.envs.wrappers.resize import Resize
from environments.garage.envs.wrappers.stack_frames import StackFrames

__all__ = [
    'AtariEnv', 'ClipReward', 'EpisodicLife', 'FireReset', 'Grayscale',
    'MaxAndSkip', 'Noop', 'PixelObservationWrapper', 'Resize', 'StackFrames'
]
