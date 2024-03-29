# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory & Planning Game environment from Rapid Task-Solving in Novel Environments (Ritter et al., 2021)"""
import string

# import dm_env
import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from io import BytesIO

from gym import spaces


class MemoryPlanningGame(gym.Env):
  """Memory & Planning Game environment."""

  ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Collect']
  NUM_ACTIONS = len(ACTION_NAMES)
  DIRECTIONS = [
      (0, 1),   # Up
      (0, -1),  # Down
      (-1, 0),  # Left
      (1, 0),   # Right
      (0, 0),   # Collect
  ]

  def __init__(self,
               maze_size=4,
               max_episode_steps=100,
               target_reward=1.,
               per_step_reward=0.,
               random_respawn=False,
               show_xy=False,
               show_goal_xy=False,
               just_show_whether_at_goal=False,
               seed=None):
    """The Memory & Planning Game environment.

    Args:
      maze_size: (int) size of the maze dimension.
      max_episode_steps: (int) number of steps per episode.
      target_reward: (float) reward value of the target.
      per_step_reward: (float) reward/cost of taking a step.
      random_respawn: (bool) whether the agent respawns in a random location
        upon collecting the goal.
      seed: (int or None) seed for random number generator.
    """
    self._maze_size = maze_size
    self._num_labels = maze_size * maze_size
    # The graph itself is the same across episodes, but the node labels will be
    # randomly sampled in each episode.
    self._graph = nx.grid_2d_graph(
        self._maze_size, self._maze_size, periodic=True)
    self._max_episode_steps = max_episode_steps
    self._target_reward = target_reward
    self._per_step_reward = per_step_reward
    self._random_respawn = random_respawn
    self._rng = np.random.RandomState(seed)
    self.show_xy = show_xy
    self.show_goal_xy = show_goal_xy
    self.just_show_whether_at_goal = just_show_whether_at_goal

    if just_show_whether_at_goal:
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
    else:
        # One hot goal and position are default obs, but can also append xy of position or goal to make task easier
        obs_len = 2*self._num_labels
        obs_high = 1
        if show_xy or show_goal_xy:
            obs_high = self._maze_size - 1
        if show_xy:
            obs_len += 2
        if show_goal_xy:
            obs_len += 2
        self.observation_space = spaces.Box(low=0, high=obs_high, shape=(obs_len,))
    self.action_space = spaces.Discrete(self.NUM_ACTIONS)

    self.reset()
    self.task_dim = len(self.get_task())
    self.num_tasks = self._num_labels

  def _one_hot(self, node):
    one_hot_vector = np.zeros([self._num_labels], dtype=np.int32)
    one_hot_vector[self._labels[node]] = 1
    return one_hot_vector

  def step(self, action):
    action = np.array(action).item()

    # If previous step was the last step of an episode, reset.
    if self._needs_reset:
      return self.reset()

    # Increment step count and check if it's the last step of the episode.
    self._episode_steps += 1
    if self._episode_steps >= self._max_episode_steps:
      self._needs_reset = True
      done = True
    else:
      done = False

    # Recompute agent's position given the selected action.
    direction = self.DIRECTIONS[action]
    self._position = tuple(
        (np.array(self._position) + np.array(direction)) % self._maze_size)
    self._previous_action = self.ACTION_NAMES[action]

    # Get reward if agent is over the goal location and the selected action is
    # `collect`.
    if self._position == self._goal and self.ACTION_NAMES[action] == 'Collect':
      reward = self._target_reward
      self._set_new_goal()
    else:
      reward = self._per_step_reward
    self._episode_reward += reward

    return self.get_obs(), reward, done, {'task': self.get_task()}
  
  def get_obs(self):

    if self.just_show_whether_at_goal:
        at_goal = (self._position == self._goal)
        return np.array([1 if at_goal else 0], dtype=np.int32)

    obs = np.concatenate((self._one_hot(self.position), self._one_hot(self.goal)), axis=-1)
    
    if self.show_xy or self.show_goal_xy:
        obs = obs.astype(np.float32)
    if self.show_xy:
        obs = np.concatenate((obs, np.array(self.position, dtype=np.float32)), axis=-1)
    if self.show_goal_xy:
        obs = np.concatenate((obs, np.array(self.goal, dtype=np.float32)), axis=-1)
    
    return obs

  # Here is the original code, replaced by get_obs() above:
  # def _observation(self):
  #   return {
  #       'position': np.array(self._one_hot(self.position), dtype=np.int32),
  #       'goal': np.array(self._one_hot(self.goal), dtype=np.int32),
  #   }

  # Here is the original code, replaced in __init__ above:
  # def observation_spec(self):
  #   return {
  #       'position': dm_env.specs.Array(
  #           shape=(self._num_labels,), dtype=np.int32, name='position'),
  #       'goal': dm_env.specs.Array(
  #           shape=(self._num_labels,), dtype=np.int32, name='goal'),
  #   }
  # def action_spec(self):
  #   return dm_env.specs.DiscreteArray(self.NUM_ACTIONS)
  # def take_random_action(self):
  #   return self.step(self._rng.randint(self.NUM_ACTIONS))

  def reset_task(self, task=None):
    """In this env, the env resets the task itself based on the agent's position, so just reset the env"""
    self.reset()
    return np.array(self._goal)

  def get_task(self):
    return np.array(self._goal)

  def reset(self):
    self._previous_action = ''
    self._episode_reward = 0.
    self._episode_steps = 0
    self._needs_reset = False
    random_labels = self._rng.permutation(self._num_labels)
    self._labels = {n: random_labels[i]
                    for i, n in enumerate(self._graph.nodes())}
    self._respawn()
    self._set_new_goal()
    return self.get_obs()

  def _respawn(self):
    random_idx = self._rng.randint(self._num_labels)
    self._position = list(self._graph.nodes())[random_idx]

  def _set_new_goal(self):
    if self._random_respawn:
      self._respawn()
    goal = self._position
    while goal == self._position:
      random_idx = self._rng.randint(self._num_labels)
      goal = list(self._graph.nodes())[random_idx]
    self._goal = goal

  @property
  def position(self):
    return self._position

  @property
  def goal(self):
    return self._goal

  @property
  def previous_action(self):
    return self._previous_action

  @property
  def episode_reward(self):
    return self._episode_reward

  def render(self, mode='rgb_array'):
    fig, ax = self.draw_maze()

    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()

    # Open the image from the buffer and convert to RGB
    buffer.seek(0)
    image = Image.open(buffer)
    image_rgb = np.array(image.convert('RGB'))

    return image_rgb

  def draw_maze(self, ax=None):
    if ax is None:
      plt.figure()
      ax = plt.gca()
    node_positions = {(x, y): (x, y) for x, y in self._graph.nodes()}
    letters = string.ascii_uppercase + string.ascii_lowercase
    labels = {n: letters[self._labels[n]] for n in self._graph.nodes()}
    node_list = list(self._graph.nodes())
    colors = []
    for n in node_list:
      if n == self.position:
        colors.append('lightblue')
      elif n == self.goal:
        colors.append('lightgreen')
      else:
        colors.append('pink')
    nx.draw(self._graph, pos=node_positions, nodelist=node_list, ax=ax,
            node_color=colors, with_labels=True, node_size=200, labels=labels)
    ax.set_title('{}\nEpisode reward={:.1f}'.format(
        self.previous_action, self.episode_reward))
    ax.margins(.1)
    return plt.gcf(), ax