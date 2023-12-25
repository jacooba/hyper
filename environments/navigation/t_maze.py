'''
An Env designed to test different aspects of long term memory.
Originally from AMRL paper
Github: https://github.com/jacooba/AMRL-ICLR2020/
Paper: https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete, Box, Tuple
from gym.envs.registration import EnvSpec
from gym.utils import seeding

import numpy as np
import torch

TMAZE_ENV_KEY = "tmaze-v0"

class Actions():
    """Enum class to define actions"""
    LEFT, RIGHT, UP, DOWN = range(4) # Must be permutations of [0...num_actions-1]

class Indicator_Dir():
    """Enum class to define the values (directions) of the longterm indicator"""
    UP, DOWN = Actions.UP, Actions.DOWN # For making equality convenient

class Check_Dir():
    """Enum class to define the values (directions) of the intermediate indicators"""
    UP, DOWN = Actions.UP, Actions.DOWN # For making equality convenient

class Obs():
    def __init__(self, check_up, check_down):
        """Class to define observations"""
        self.START         = [1,0,0]
        self.START_UP      = [1,1,0]
        self.START_DOWN    = [1,-1,0]
        self.END           = [0,0,1]
        self.END_UP        = [0,1,1]
        self.END_DOWN      = [0,-1,1]
        self.MIDDLE        = [0,0,0]
        self.MIDDLE_UP     = [0,1,0]
        self.MIDDLE_DOWN   = [0,-1,0]
        # The following indicators are appeneded to above if self.intermediate_indicators:
        self.CHECK_UP   = check_up
        self.CHECK_DOWN = check_down
        self.END_CHECK  = 0 # Sepcial case for the end, if there is not be an indicator there
        self.CHECK2OBS  = {
            Check_Dir.UP: self.CHECK_UP, 
            Check_Dir.DOWN: self.CHECK_DOWN,
        }
        self.OBS2CHECK = { # Useful for testing
            self.CHECK_UP: Check_Dir.UP,
            self.CHECK_DOWN: Check_Dir.DOWN,
        }

        self.OBS_LIST = [self.START, self.START_UP, self.START_DOWN, self.END, self.END_UP, self.END_DOWN, 
                    self.MIDDLE, self.MIDDLE_UP, self.MIDDLE_DOWN]
        self.CHECK_LIST = [[self.CHECK_UP], [self.CHECK_DOWN], [self.END_CHECK]]

    def get_observation_space(self, intermediate_indicators, pos_enc, wave_encoding_len, timeout, num_indicators_components=None):
        if intermediate_indicators:
            assert num_indicators_components is not None, num_indicators_components
            main_obs_low = np.array(self.OBS_LIST).min(axis=0)
            main_obs_hi = np.array(self.OBS_LIST).max(axis=0)
            indicator_component_low = np.array(self.CHECK_LIST).min(axis=0)
            indicator_component_hi = np.array(self.CHECK_LIST).max(axis=0)
            indicator_low = np.broadcast_to(indicator_component_low, (num_indicators_components))
            indicator_hi = np.broadcast_to(indicator_component_hi, (num_indicators_components))
            low = np.concatenate((main_obs_low, indicator_low), axis=-1)
            hi = np.concatenate((main_obs_hi, indicator_hi), axis=-1)
        else:
            low = np.array(self.OBS_LIST).min(axis=0)
            hi = np.array(self.OBS_LIST).max(axis=0)

        if pos_enc:
            if wave_encoding_len is None:
                enc_low = np.array([0])
                enc_hi = np.array([timeout])
            else:
                enc_low = np.array([-1 for _ in range(wave_encoding_len)])
                enc_hi = np.array([1 for _ in range(wave_encoding_len)])
            low = np.concatenate((low, enc_low), axis=-1)
            hi = np.concatenate((hi, enc_hi), axis=-1)

        return Box(low, hi, dtype=np.float32)

class TMaze(gym.Env):
    """
    T-maze in which an indicator along a corridor corresponds to the goal location
    at the end of the corridor.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:
            allow_left (bool): Whether the agent should be able to step left (backwards)
            timeout (int): How many steps the agent is allowed to take before the episode terminates
            timeout_reward (float): The reward the agent receives form a timeout
            maze_length (int): The length of the maze (number of steps to solve optimally)
            indicator_pos (int): The location [0,maze_length) to place the indicator meant to be remembered
            success_reward (float): The reward for choosing the correct action (up or down) at the end of the maze
            fail_reward (float): The reward for choosing the incorrect action (up or down) at the end of the maze
            check_reward (float): If there are intermediate checks (tasks), the reward per step for getting it correct
            persistent_reward (float): A reward given per timestep
            force_final_decision (bool): Whether the agent must choose up or down at the end of the maze
            force_right (bool): Whether the agent goes to the right in the corridor (progreses) regardless of action
            intermediate_checks (bool): Whether there is an additional task of reproducing an observation at each step
            intermediate_indicators(bool): Whether there are additional observations (bitsup or down) at each step.
            reset_intermediate_indicators (bool): Whether the intermediate observation reset between episodes
            final_intermediate_indicator (bool): Whether there should be an intermediate observation at the end.
            per_step_reset (bool): Whether the intermediate obseravtoins reset between steps
            flipped_indicator_pos (int): If specified, an indicator will be placed here repreenting the opposite
                                         direction as that of the standard long term indicator
            wave_encoding_len (int or None): If specified and pos_enc is True, then the timestep, encoded using
                                             sine and cosine waves with this number of dimensions will be added to 
                                             the observation
            pos_enc (bool): Whether to outpute the timestep as part of the observation
            correlated_indicator_pos (int or None): If specified, another long-term indicator will be placed here,
                                                    and the agent must go up at the end iff the first indicator
                                                    and this indicator together occur with the pattern: UP, DOWN
            check_up (int): An int used to encode an intermedaite check "up" observation (per dimension)
            check_down (int): An int used to encode an intermedaite check "down" observation (per dimension)
            maze_length_upper_bound (int or None): If specified, a random maze length will be sampled
                                                   uniformly at random between maze_length and maze_length_upper_bound,
                                                   to test generalization
    """

    def __init__(self, config):
        required_args = set(["allow_left",
                              "timeout",
                              "timeout_reward",
                              "maze_length",
                              "indicator_pos",
                              "success_reward",
                              "fail_reward",
                              "check_reward",
                              "persistent_reward",
                              "force_final_decision",
                              "force_right",
                              "intermediate_checks",
                              "intermediate_indicators",
                              "reset_intermediate_indicators",
                              "final_intermediate_indicator",
                              "per_step_reset",
                              "flipped_indicator_pos",
                              "wave_encoding_len",
                              "pos_enc",
                              "correlated_indicator_pos",
                              "check_up",
                              "check_down",
                              "maze_length_upper_bound",])
        given_args = set(config.keys())
        self.OBS = Obs(check_up=config["check_up"], check_down=config["check_down"])
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)
        self.force_final_decision = config["force_final_decision"]
        self.force_right = config["force_right"]
        self.allow_left = config["allow_left"]
        self.intermediate_checks = config["intermediate_checks"]
        self.intermediate_indicators = config["intermediate_indicators"]
        self.reset_intermediate_indicators = config["reset_intermediate_indicators"]
        self.final_intermediate_indicator = config["final_intermediate_indicator"]
        self.per_step_reset = config["per_step_reset"]
        self.wave_encoding_len = config["wave_encoding_len"]
        self.pos_enc = config["pos_enc"]
        assert not (self.force_right and self.allow_left), "Cannot force right action and allow left action"
        assert not (self.force_right and self.intermediate_checks), "Cannot force right action and do intermediate checks"
        assert not (self.intermediate_checks and not self.intermediate_indicators), "Intermediate indicators required to do intermediate checks"
        self.timeout = config["timeout"]
        self._max_episode_steps = self.timeout+1
        self.timeout_reward = config["timeout_reward"]
        self.success_reward = config["success_reward"]
        self.fail_reward = config["fail_reward"]
        self.check_reward = config["check_reward"]
        self.persistent_reward = config["persistent_reward"]
        self.maze_len = config["maze_length"]
        assert self.maze_len >= 1, self.maze_len
        self.maze_length_upper_bound = config["maze_length_upper_bound"]
        if self.maze_length_upper_bound is not None:
            self.maze_length_lower_bound = self.maze_len
            assert self.maze_length_upper_bound >= self.maze_len, (self.maze_length_upper_bound, self.maze_len)
        self.indicator_pos = config["indicator_pos"]
        assert self.indicator_pos >= 0 and self.indicator_pos <= self.maze_len-1, self.indicator_pos
        
        self.flipped_indicator_pos = config["flipped_indicator_pos"]
        self.correlated_indicator_pos = config["correlated_indicator_pos"]
        indicator_at_endpoint = (self.indicator_pos == 0) or (self.indicator_pos == (self.maze_len-1))
        if self.flipped_indicator_pos is not None:
            assert self.correlated_indicator_pos is None, "Cannot have correlated indicator and flipped indicator currently"
            flipped_indicator_at_endpoint = (self.flipped_indicator_pos == 0) or (self.flipped_indicator_pos == (self.maze_len-1))
            assert not (indicator_at_endpoint or flipped_indicator_at_endpoint), \
                   "This is probably won't test the order dependence I want, so not implemented."
            assert not (self.flipped_indicator_pos == self.indicator_pos), "Cannot have both long term indicators in the same location"
        if self.correlated_indicator_pos is not None:
            correlated_indicator_at_endpoint = (self.correlated_indicator_pos == 0) or (self.correlated_indicator_pos == (self.maze_len-1))
            assert not (indicator_at_endpoint or correlated_indicator_at_endpoint), \
                   "This is probably won't test the order dependence I want, so not implemented."
            assert not (self.correlated_indicator_pos == self.indicator_pos), "Cannot have both long term indicators in the same location"

        # This used to be editable, but to avoid Tuple action space, it is hardcoded
        self.num_indicators_components = 1

        self.action_space = Discrete(4) # Directions in maze
        self.observation_space = self.OBS.get_observation_space(self.intermediate_indicators, self.pos_enc, self.wave_encoding_len, self.timeout,
                                                           num_indicators_components=self.num_indicators_components)
        self._spec = EnvSpec(TMAZE_ENV_KEY)

        self.seed()
        self.reset_task(None, return_obs=False) # obs not yet initialized until next line
        self.reset()

        self.task_dim = len(self.get_task())
        self.num_tasks = 2 if self.secondary_indicator is None else 4

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_random_indicator(self):
        return self.np_random.choice([Indicator_Dir.UP, Indicator_Dir.DOWN], p=[0.5, 0.5])

    def get_task(self):
        return np.array([self.indicator]) if self.secondary_indicator is None else np.array([self.indicator, self.secondary_indicator]) 

    def task_to_id(self, tasks):
        tasks = tasks.to(torch.int64)
        if self.secondary_indicator is None:
            ids = torch.where(tasks == Indicator_Dir.UP, torch.ones_like(tasks), torch.zeros_like(tasks))
        else:
            ids = torch.where(tasks[:,0] == Indicator_Dir.UP, torch.ones_like(tasks[:,0]), torch.zeros_like(tasks[:,0]))
            ids = ids + 2*torch.where(tasks[:,1] == Indicator_Dir.UP, torch.ones_like(tasks[:,1]), torch.zeros_like(tasks[:,1]))
        return ids

    def reset(self):
        self.timestep = 0
        self.cur_pos = 0

        if self.intermediate_indicators:
            random = self.np_random if self.reset_intermediate_indicators else np.random.RandomState(0)
            self.checks = random.choice([Check_Dir.UP, Check_Dir.DOWN], 
                                         size=(self.maze_len, self.num_indicators_components), p=[0.5, 0.5])
            if not self.final_intermediate_indicator:
                self.checks = list(self.checks)
                self.checks[-1] = None

        return self.get_obs()

    def reset_task(self, task, return_obs=True):
        if self.maze_length_upper_bound is not None:
            self.maze_len = self.np_random.randint(low=self.maze_length_lower_bound, high=self.maze_length_upper_bound+1)
        self.indicator = self.get_random_indicator() if task is None else task[0]
        assert self.indicator in [Indicator_Dir().UP, Indicator_Dir().DOWN]
        
        self.secondary_indicator_pos = None
        self.secondary_indicator = None
        if self.flipped_indicator_pos is not None:
            self.secondary_indicator_pos = self.flipped_indicator_pos
            self.secondary_indicator = Indicator_Dir.UP if self.indicator == Indicator_Dir.DOWN else Indicator_Dir.DOWN
        elif self.correlated_indicator_pos is not None:
            self.secondary_indicator_pos = self.correlated_indicator_pos
            self.secondary_indicator = self.get_random_indicator() if task is None else task[1] # This is correlated with goal in that DD,UU,DU -> D; UD -> U

        return self.get_obs() if return_obs else None

    def add_check(self, obs):
        """ Adds the check to the end of obs if self.intermediate_indicators """
        
        if not self.intermediate_indicators:
            return obs

        if self.get_cur_check() is None:
            assert self.cur_pos == self.maze_len-1, self.cur_pos
            check_obs = [self.OBS.END_CHECK for _ in range(self.num_indicators_components)]
        else:
            check_obs = [self.OBS.CHECK2OBS[c] for c in self.get_cur_check()]

        return obs + check_obs

    def pos_wave_encoding(self, p, l, c=10000):
        """ Calculates a positional encoding of length l for position p inline with https://arxiv.org/pdf/1706.03762 """
        enc = [None for _ in range(l)]
        for i in range(l):
            v = p / (c**(2*i / l))
            enc[i] = np.sin(v) if (i%2) == 0 else np.cos(v)
        return enc

    def add_positional_encoding(self, obs):
        """ Adds a positional encoding to the input """
        if not self.pos_enc:
            return obs
        return obs + (self.pos_wave_encoding(self.timestep, self.wave_encoding_len, self.timeout) if (self.wave_encoding_len is not None) \
               else [self.timestep])

    def get_obs_without_check(self):
        """Returns the correct current observation assuming not self.intermediate_indicators."""
        
        if self.cur_pos == 0:
            # Start
            if self.indicator_pos != self.cur_pos: # Indicator not here
                return self.OBS.START
            if self.indicator == Indicator_Dir.UP: # Indicator up
                return self.OBS.START_UP
            assert self.indicator == Indicator_Dir.DOWN, self.indicator
            return self.OBS.START_DOWN # Indicator down

        if self.cur_pos == self.maze_len-1:
            # End
            if self.indicator_pos != self.cur_pos: # Indicator not here
                return self.OBS.END
            if self.indicator == Indicator_Dir.UP: # Indicator up
                return self.OBS.END_UP
            assert self.indicator == Indicator_Dir.DOWN, self.indicator
            return self.OBS.END_DOWN # Indicator down

        assert self.cur_pos > 0 and self.cur_pos < self.maze_len-1, self.cur_pos
        # Middle
        if (self.indicator_pos != self.cur_pos) and (self.secondary_indicator_pos != self.cur_pos): # Indicator not here
            return self.OBS.MIDDLE # Regular middle (no indicator)
        if self.indicator_pos == self.cur_pos:
            ind = self.indicator
        elif self.secondary_indicator_pos == self.cur_pos:
            ind = self.secondary_indicator
        if ind == Indicator_Dir.UP: # Indicator up
            return self.OBS.MIDDLE_UP
        assert ind == Indicator_Dir.DOWN, ind
        return self.OBS.MIDDLE_DOWN # Indicator down  

    def get_obs(self):
        """Returns the correct current observation."""
        return np.array(self.add_positional_encoding(self.add_check(self.get_obs_without_check())))

    def get_cur_check(self):
        return self.checks[self.cur_pos]

    def move_right(self):
        self.cur_pos += 1
        self.cur_pos = min(self.cur_pos, self.maze_len-1)

    def move_left(self):
        self.cur_pos -= 1
        self.cur_pos = max(self.cur_pos, 0)

    def reset_cur_check(self):
        if self.checks[self.cur_pos] is not None:
            self.checks[self.cur_pos] = self.np_random.choice([Check_Dir.UP, Check_Dir.DOWN], 
                                                              size=(self.num_indicators_components), p=[0.5, 0.5])

    def step(self, action):
        a = action

        # timeout if necessary
        if self.timeout is not None and self.timestep > self.timeout-1:
            if self.reset_intermediate_indicators and self.per_step_reset:
                self.reset_cur_check()
            return self.get_obs(), self.timeout_reward, True, {}

        # options that force decions (by changing action taken)
        if self.force_final_decision and (self.cur_pos == self.maze_len-1):
            if a == Actions.LEFT:
                a = Actions.UP
            elif a == Actions.RIGHT:
                a = Actions.DOWN
        if self.force_right and (self.cur_pos < self.maze_len-1):
            a = Actions.RIGHT

        # check the direction of a and respond appropriately
        done = False
        reward = self.persistent_reward
        moved_left = False
        if a == Actions.LEFT:
            if self.allow_left:
                self.move_left()
                moved_left = True
        elif a == Actions.RIGHT:
            if not self.intermediate_checks:
                self.move_right()
        else:
            assert a in [Actions.UP, Actions.DOWN]
            if self.cur_pos == self.maze_len-1: # At end
                # Tell whether successful
                if self.correlated_indicator_pos is None:
                    success = (a == self.indicator)# Only one indicator and agent is correct
                else:
                    indicator_pattern = (self.indicator, self.secondary_indicator) # DD,UU,DU -> D; UD -> U
                    up_patterns = [(Actions.UP, Actions.DOWN)]
                    down_patterns = [(Actions.DOWN, Actions.DOWN), (Actions.UP, Actions.UP), (Actions.DOWN, Actions.UP)]
                    assert indicator_pattern in (up_patterns + down_patterns), indicator_pattern
                    success = ((a == Actions.UP) and (indicator_pattern in up_patterns)) \
                          or ((a == Actions.DOWN) and (indicator_pattern in down_patterns))
                # Give reward and addign done
                reward += self.success_reward if success else self.fail_reward
                done = True
        
        # Deal with intermediate checks
        # Only do this in the case that you havent already moved left (which takes precedence) and not at end
        # (If you could get intermediate reward at end, you might prefer to not end episode.)
        if self.intermediate_checks and (self.cur_pos < self.maze_len-1) and not moved_left:
            cur_check = self.get_cur_check()
            assert len(cur_check) == 1
            correct = cur_check[0] == action
            if correct:
                check_r = self.check_reward
                reward += check_r
                self.move_right()

        # Update timestep (and intermediate indicator) and return
        self.timestep += 1
        if self.intermediate_indicators and self.reset_intermediate_indicators and self.per_step_reset:
            self.reset_cur_check()
        return self.get_obs(), reward, done, {'task': self.get_task()}

    def render(self, mode='human'):
        lines = [['x' for _ in range(self.maze_len+2)] for _ in range(5)]
        lines[2][1:-1] = [' ' for _ in range(self.maze_len)]
        lines[2][self.indicator_pos+1] = 'I'

        if self.intermediate_indicators:
            for i in range(len(self.checks)):
                lines[1][i+1] = "u" if self.checks[i][0] == Actions.UP else "d"

        if self.indicator == Indicator_Dir.UP:
            lines[1][-2] = 'g'
            lines[3][-2] = ' '
        else:
            lines[1][-2] = ' '
            lines[3][-2] = 'g'

        lines[2][self.cur_pos+1] = 'a'

        print()
        print('\n'.join([''.join(line) for line in lines]))  

if __name__ == "__main__":
    # run env repl with config:
    example_TLN_config = {
        "check_up": 1, # Observation for check / noise pointed up
        "check_down": -1,
        "maze_length_upper_bound": None,
        "pos_enc": False,
        "wave_encoding_len": None, #3 # Can be null for no not wave-based encoding
        "intermediate_checks": True,
        "intermediate_indicators": True, # Whether there are intermediate indicators. (Even if no checks, will increase action dimension)
        "reset_intermediate_indicators": True, # whether the intermediate indicators change from episode to episode
        "per_step_reset": True,
        "final_intermediate_indicator": True, # Whether or not there is an intermediate indicator at then end
        "check_reward": 0.1,
        "allow_left": False,
        "force_final_decision": True, # Whether to force the agent to move up or down at end
        "force_right": False, # Whether to force the agent to move right (not at end and not for checks)
        "timeout": 150, # Max steps allowed or null
        "timeout_reward": 0,
        "maze_length": 4,
        "indicator_pos": 0,
        "flipped_indicator_pos": None, #can be null for no duplicate flipped indicator
        "correlated_indicator_pos": None, #can be null for no correlated indicator
        "success_reward": 4.0,
        "fail_reward": -3.0,
        "persistent_reward": 0.0, # Reward given per time step
    }

    env = TMaze(example_TLN_config)

    num_meta_ep = 2
    num_inner_ep = 2
    for i in range(num_meta_ep):
        env.reset_task(task=None)
        print("Meta Episode:", i+1)
        for j in range(num_inner_ep):
            obs = env.reset()
            print("\nobs:", obs)
            env.render()
            print("inner ep:", j+1)
            done = False
            while not done:
                action_map = {"a": Actions().LEFT, 
                              "d": Actions().RIGHT, 
                              "w": Actions().UP, 
                              "s": Actions().DOWN}
                action = None
                while action not in action_map:
                    print("Action?")
                    print("actions allowed: w,a,s,d")
                    action = input()
                obs, reward, done, info = env.step(action_map[action])
                env.render()
                print("obs:", obs)
                print("done:", done)
                print("info:", info)
                print("reward:", reward)



