import gym
import numpy as np
from gym import spaces
from gym.envs.registration import load
from environments.garage import Wrapper, EnvStep, EnvSpec
import random
import akro
import gc

import torch
import torch.nn.functional as F



class MetaWorldMultiEnvWrapper(Wrapper):  # Adapted from Garage's MultiEnvWrapper
    """A wrapper class to handle multiple environments of MetaWorld.
    This wrapper adds an integer 'task_id' to env_info every timestep.
    Args:
        envs (list(Environment)):
            A list of objects implementing Environment.
        sample_strategy (function(int, int)):
            Sample strategy to be used when sampling a new task.
        mode (str): A string from 'vanilla`, 'add-onehot' and 'del-onehot'.
            The type of observation to use.
            - 'vanilla' provides the observation as it is.
              Use case: metaworld environments with MT* algorithms,
                        gym environments with Task Embedding.
            - 'add-onehot' will append an one-hot task id to observation.
              Use case: gym environments with MT* algorithms.
              NOTE: currently shuffling will mess this up. ATM this returns the index in the shuffled array! get_task is correct for id.
            - 'del-onehot' assumes an one-hot task id is appended to
              observation, and it excludes that.
              Use case: metaworld environments with Task Embedding.
        env_names (list(str)): The names of the environments corresponding to
            envs. The index of an env_name must correspond to the index of the
            corresponding env in envs. An env_name in env_names must be unique.
    """

    def __init__(self,
                 task_sampler,
                 n_tasks_train,
                 n_tasks_test,
                 mode='vanilla',
                 env_names=None,
                 train=True):
        assert mode in ['vanilla', 'add-onehot', 'del-onehot']

        self.n_tasks_train = n_tasks_train
        self.n_tasks_test = n_tasks_test
        self.n_tasks_total = n_tasks_train+n_tasks_test # This is total, between training and test modes!
        self._num_tasks_here = n_tasks_train if train else n_tasks_test # num tasks in this mode!
        self._active_task_index = 0
        self._mode = mode
        self.task_sampler = task_sampler
        self.train = train

        self._task_envs = None # replaced on next line
        self._active_task_index = self.sample_strategy(self._num_tasks_here-1) # sample new envs for each task
        self._env = self.get_new_env()

        super().__init__(self._env)

        if env_names is not None:
            assert isinstance(env_names, list), 'env_names must be a list'

        self._env_names = env_names

        # This attribute is needed here and potentially in other code
        # self.n_tasks_total or self.n_tasks_train are both reasonable, depending on whether you will include test ids in the one-hot encoding
        # see get_task() below.
        self.num_tasks = self.n_tasks_train
        self.task_dim = self.num_tasks + 3 # + 3 for goal location

    @property
    def observation_space(self):
        """Observation space.
        Returns:
            akro.Box: Observation space.
        """
        if self._mode == 'vanilla':
            return self._env.observation_space
        elif self._mode == 'add-onehot':
            task_lb, task_ub = self.task_space.bounds
            env_lb, env_ub = self._env.observation_space.bounds
            return akro.Box(np.concatenate([env_lb, task_lb]),
                            np.concatenate([env_ub, task_ub]))
        else:  # self._mode == 'del-onehot'
            env_lb, env_ub = self._env.observation_space.bounds
            num_tasks = self._num_tasks_here
            return akro.Box(env_lb[:-num_tasks], env_ub[:-num_tasks])

    @property
    def spec(self):
        """Describes the action and observation spaces of the wrapped envs.
        Returns:
            EnvSpec: the action and observation spaces of the
                wrapped environments.
        """
        return EnvSpec(action_space=self.action_space,
                       observation_space=self.observation_space,
                       max_episode_length=self._env.spec.max_episode_length)

    # Jake: This had to be an actual attribute to work with other code...
    # @property
    # def num_tasks(self):
    #     """Total number of tasks.
    #     Returns:
    #         int: number of tasks.
    #     """
    #     # self.n_tasks_total or self.n_tasks_train, depending on whether you will include test ids in the one-hot encoding
    #     # see get_task() below.
    #     return self.n_tasks_train

    @property
    def task_space(self):
        """Task Space.
        Returns:
            akro.Box: Task space.
        """
        one_hot_ub = np.ones(self.num_tasks)
        one_hot_lb = np.zeros(self.num_tasks)
        ub = np.concatenate([one_hot_ub,  1*np.ones(3)]) # length 3 goal will be added to task
        lb = np.concatenate([one_hot_lb, -1*np.ones(3)])
        return akro.Box(lb, ub)

    @property
    def active_task_index(self):
        """Index of active task env.
        Returns:
            int: Index of active task.
        """
        if hasattr(self._env, 'active_task_index'):
            return self._env.active_task_index
        else:
            return self._active_task_index

    def all_task_ids(self):
        return list(range(self.n_tasks_train)) if self.train else list(range(self.n_tasks_train, self.n_tasks_total))

    def sample_strategy(self, task_index):
        """ Samples the next task. Currently, it's set to do it in order and sample new tasks if all have been done
            Saves memory by closing the previous task environment before starting a new one
        """

        if task_index + 1 == self._num_tasks_here:
            del self._task_envs
            gc.collect()
            self._task_envs = [env_up for env_up in self.task_sampler.sample(self._num_tasks_here)]
            assert len(self.all_task_ids()) == len(self._task_envs), (len(self.all_task_ids()) == len(self._task_envs))
            self._shuffled_env_id_tups = list(zip(self._task_envs, self.all_task_ids()))
            random.shuffle(self._shuffled_env_id_tups)

        task_index = (1 + task_index) % self._num_tasks_here

        return task_index

    def get_new_env(self):
        env_f, _ = self._shuffled_env_id_tups[self._active_task_index]
        env = env_f()
        # Hack to get this reset_task working
        # env.unwrapped.reset_task = None # env.unwrapped.reset ... I don't think this should be called. None makes sure it is caught here.
        env.unwrapped._max_episode_steps = env.max_path_length
        return env

    def get_task(self):
        task_one_hot = np.zeros((self.num_tasks,))
        if self.train:
            _, task_id = self._shuffled_env_id_tups[self._active_task_index]
            task_one_hot[task_id] = 1 # in torch: F.one_hot(task_id, num_classes=self.num_tasks)
        return np.concatenate([task_one_hot, self._env.unwrapped._get_pos_goal()]).astype(np.float32)

    def reset_task(self):
        """Sample new task and call reset on new task env. (Originally "reset" function from MultiEnvWrapper)
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        self._env.close() # close current task
        self._active_task_index = self.sample_strategy(self.active_task_index) # sample new task
        self._env = self.get_new_env()

        obs, episode_info = self._env.reset()

        if self._mode == 'vanilla':
            pass
        elif self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        else:  # self._mode == 'del-onehot'
            obs = obs[:-self._num_tasks_here]

        return obs, episode_info

    def reset(self):
        """Resets the current task """
        # Reset NOT the MultiEnvWrapper -- we don't want to sample a new task
        old_task = self.get_task()
        obs, episode_info = self._env.reset()
        new_task = self.get_task()
        assert (old_task == new_task).all(), (old_task, new_task)
        return obs

    def sup_step(self, action):
        """Step the active task env. (Original function from MultiEnvWrapper)
        Args:
            action (object): object to be passed in Environment.reset(action)
        Returns:
            EnvStep: The environment step resulting from the action.
        """
        es = self._env.step(action)

        if self._mode == 'add-onehot':
            obs = np.concatenate([es.observation, self._active_task_one_hot()])
        elif self._mode == 'del-onehot':
            obs = es.observation[:-self._num_tasks_here]
        else:  # self._mode == 'vanilla'
            obs = es.observation

        env_info = es.env_info
        if 'task_id' not in es.env_info:
            env_info['task_id'] = self._active_task_index
        if self._env_names is not None:
            env_info['task_name'] = self._env_names[self._active_task_index]

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=es.reward,
                       observation=obs,
                       env_info=env_info,
                       step_type=es.step_type)

    def step(self, action):
        data = self.sup_step(action)
        # rew = float(info['success'])
        # return data.observation, data.reward, data.env_info['success'], data.env_info
        return data.observation, data.reward, data.env_info['success'], data.env_info

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            try:
                # In case that the environment is not closed (should not require closing)
                env.close()
            except:
                pass

    def _active_task_one_hot(self):
        """One-hot representation of active task.
        Returns:
            numpy.ndarray: one-hot representation of active task
        """
        one_hot = np.zeros(self.task_space.shape)
        index = self.active_task_index or 0
        one_hot[index] = self.task_space.high[index]
        return one_hot

