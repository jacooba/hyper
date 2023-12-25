'''
The environment used in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
Code: https://github.com/jacooba/AMRL-ICLR2020
'''


import argparse
import numpy as np
import time
import random
import os
import sys
import glob
from PIL import Image
from shutil import copytree, rmtree

# gym imports
import gym
from gym.spaces import Discrete, Box, Tuple
from gym.utils import seeding
from gym.spaces import Discrete, Box

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DOWNSAMPLE_SZ = 10 # Note 42 in original
LOW_RES_STR = str(DOWNSAMPLE_SZ)+"_resolution"
LOW_RES_DIR = "./mine_maze_data/"+LOW_RES_STR

# Default configs for REPL in main()
CACHED_CONFIG = {
    "num_rooms": 16,
    "multi_step_indicator": True,
    "num_single_step_repeats": 1,
    "success_r": 4, 
    "fail_r": -3, 
    "check_success_r": 0.1, 
    "check_fail_r": 0.0,
    "reward_per_progress": 0.1,
    "timeout": 200,
    "high_res": True,
    "noise": 0.05,
}

def file_path_to_numpy_img(fp):
    return np.flip(np.array(Image.open(fp)), axis=0)

def write_obs(obs, fp=None):
    if fp is None:
        fp = 'temp_image.png'
    obs = (obs * 128) + 128 # unnormalize
    obs = np.clip(np.round(obs), 0, 256)
    obs = obs.astype(np.uint8)
    img = Image.fromarray(np.flip(obs, axis=(0)))
    save_path = os.path.join(THIS_DIR, fp)
    img.save(save_path)

class Actions():
    """Enum class to define actions"""
    LEFT, RIGHT, UP, DOWN = range(4) # Must be permutations of [0...num_actions-1]

class Indicator_Dir():
    """Enum class to define the values (directions) of the longterm indicator"""
    UP, DOWN = Actions.UP, Actions.DOWN # For making equality convenient

class MineMaze(gym.Env):
    """
    A visual maze env that is a simplified version of MineMazeFull, such that it saves stored images and 
    doesn't actually need to run Minecraft.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:
            num_rooms (int): The number of rooms in the maze (defines the length)
            multi_step_indicator (bool): Whether the indicaor is a single obseravtion or consists of a pattern
                                   of two over two steps
            num_single_step_repeats (int): If single step indicator, the number of repeats of the single observation
            success_r (float): The reward for going the right way at then end of the maze (left or right) based on the 
                        indicator
            fail_r (float): The reward for going the wrong way at then end of the maze (left or right) based on the 
                     indicator
            check_success_r (float): The reward for going the right way at the intermediate checks (left/right)
            check_fail_r (float): The reward for going the wrong way at the intermediate checks (left/right)
            reward_per_progress (float): The reward for going the right way any step pther than the checks
            timeout (int): The maximum number of steps the agent can take before terminating and receiving 0 reward
            high_res (bool): Whether or not to render observations in higher-resolution
            noise (float or None): The scale of Gaussian noise to add to the observations (or None)
        hide_signal (bool): Whether or not to hide the signal so the agent must explore
    """

    def __init__(self, config, hide_signal=False):
        required_args = set([  
            "num_rooms", 
            "multi_step_indicator", 
            "num_single_step_repeats",
            "success_r",
            "fail_r",
            "check_success_r",
            "check_fail_r",
            "reward_per_progress",
            "timeout",
            "high_res",
            "noise",
        ])
        given_args = set(config.keys())
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)

        self.hide_signal = hide_signal

        self.num_rooms = config["num_rooms"]
        self.multi_step_indicator = config["multi_step_indicator"]
        self.num_single_step_repeats = config["num_single_step_repeats"]
        self.success_r = config["success_r"]
        self.fail_r = config["fail_r"]
        self.check_success_r = config["check_success_r"]
        self.check_fail_r = config["check_fail_r"]
        self.reward_per_progress = config["reward_per_progress"]
        self.timeout = config["timeout"]
        self._max_episode_steps = self.timeout+1
        self.noise = config["noise"] # Magnitude of Gaussian noise to add or None

        # Deal with action spaces
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(DOWNSAMPLE_SZ*DOWNSAMPLE_SZ*3,))

        self.task_dim = 1

        len_phase0 = 2 if self.multi_step_indicator else self.num_single_step_repeats
        self.phase_2_valid_xy = [set([(x, 0) for x in range(len_phase0)]), # phase 0
                                 {(-1, 1), (-1,2), (-1,3), (0,0), (0,1), (0,3), (1,1), (1,2), (1,3)}, #phase 2
                                 {(0, 0)}] # phase 1
        self.read_in_imgs(hi_res=config["high_res"])

        self.seed()
        self.reset_task(return_obs=False) # cannot compute obs until reset called once
        self.reset()

    def reset(self):
        self.room_types = self.np_random.choice([0, 1], size=(self.num_rooms,))
        self.step_num = 1 # Start at one since there is one obs given by this reset. (If timeout is 1, you get this plus terminal obs)
        self.room_num = 0
        self.agent_x = 0 # Relative to current room
        self.agent_y = 0
        self.phase = 0 # On upper platform, then in normal maze room, then at end
        return self.get_obs()

    def get_task(self):
        return np.array([self.indicator]) 

    def reset_task(self, task=None, return_obs=True):
        if task is None:
            # if multistep, this makes (r,r),(g,g),(g,r)(r,g) equally likely:
            up_down_probs = [0.25, 0.75] if self.multi_step_indicator else [0.5, 0.5]
            self.indicator = self.np_random.choice([Indicator_Dir.UP, Indicator_Dir.DOWN], p=up_down_probs)
        else:
            self.indicator = task[0]
        # Set indicator color for observation.
        # self.x_2_icolor will define both the number of steps in phase 0 (indicator phase) and their color
        if self.multi_step_indicator:
            self.x_2_icolor = ("g", "r") if self.indicator_is_up() else \
                              [("r", "g"), ("g", "g"), ("r", "r")][self.np_random.choice([0,1,2])]
        else:
            color = "g" if self.indicator_is_up() else "r"
            self.x_2_icolor = tuple([color for _ in range(self.num_single_step_repeats)])
        return self.get_obs() if return_obs else None

    def cur_room_type(self):
        return self.room_types[self.room_num]

    def read_in_imgs(self, hi_res):
        img_dir = "mine_maze_data/1000_resolution" if hi_res else "mine_maze_data/"+LOW_RES_STR
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        img_dir = os.path.join(parent_dir, img_dir)
        try:
            green_img = file_path_to_numpy_img(img_dir + "/green.png")
            red_img = file_path_to_numpy_img(img_dir + "/red.png")
            self.color_2_img = {"g": green_img, "r": red_img}
            self.end_img = file_path_to_numpy_img(img_dir + "/end.png")
            room_type_2_img_path = [glob.glob(img_dir + "/room0/" + "*.png"), glob.glob(img_dir + "/room1/" + "*.png")]
            # e.g. [{(0,0): img1, (0,1): img2}, for room type 1
            #       {(0,0): img3, (0,1): img4}] for room type 2
            self.room_type_2_xy_2_img = [{}, {}]
            for room_type in [0,1]:
                for fp in room_type_2_img_path[room_type]:
                    name, ext = os.path.basename(fp).split(".")
                    x, y = [int(coord_str) for coord_str in name.split(",")]
                    assert (x, y) in self.phase_2_valid_xy[1], (x, y)
                    img = file_path_to_numpy_img(fp)
                    self.room_type_2_xy_2_img[room_type][(x,y)] = img
        except FileNotFoundError:
            print("img dir",img_dir,"not found. Please run script with --downsample first")
            exit()

    def indicator_is_up(self):
        return self.indicator == Indicator_Dir.UP

    def get_obs(self, reshape=True):
        if self.phase == 0:
            if self.hide_signal:
                obs = np.zeros_like(self.end_img)
            else:
                color = self.x_2_icolor[self.agent_x]
                obs = self.color_2_img[color]
        elif self.phase == 1:
            obs = self.room_type_2_xy_2_img[self.cur_room_type()][self.xy()]
        else:
            assert self.phase == 2, self.phase
            obs = self.end_img
        self.last_unnormed_obs = obs
        obs = self.normalize_obs(obs)
        if self.noise is not None:
            noise = self.np_random.normal(loc=0.0, scale=self.noise, size=obs.shape)
            obs += noise
            obs = obs.clip(-1, 1)
        self.last_obs = obs
        return obs.flatten() if reshape else obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_obs(self, obs):
        obs = obs.astype(np.float32)
        return (obs - 128)/128

    def xy(self):
        return (self.agent_x, self.agent_y)

    def get_delta_x_delta_y(self, action):
        if action == 0:
            delta_x, delta_y = -1, 0
        elif action == 1:
            delta_x, delta_y =  0, 1
        else:
            assert action == 2, action
            delta_x, delta_y =  1, 0

        # Take care of movement within a phase
        if self.phase == 0:
            delta_y = 0 # Cannot move forard here
            delta_x = 0 if delta_x == -1 else delta_x # Cannot move left here
        elif self.phase == 1:
            if self.xy() == (0, 0):
                delta_x = 0 # Cannot move l/r
            elif self.xy() == (0, 1):
                delta_y = 0 # Cannot move f
            elif self.xy() == (1, 1) or self.xy() == (-1, 1) or \
                 self.xy() == (1, 2) or self.xy() == (-1, 2):
                delta_x = 0 # Cannot move l/r
            elif self.xy() == (-1, 3):
                delta_y = 0 # Cannot move f
                delta_x = 0 if delta_x == -1 else delta_x # Cannot move left here
            elif self.xy() == (1, 3):
                delta_y = 0 # Cannot move f
                delta_x = 0 if delta_x == 1 else delta_x # Cannot move right here
            elif self.xy() == (0, 3):
                delta_x = 0 # Cannot move l/r 
            else:
                assert self.xy() not in self.phase_2_valid_xy[1], self.xy()
                raise ValueError("Agent pos: "+str(self.xy())+" was not dealt with properly")
        else:
            assert self.phase == 2, self.phase
            delta_y = 0 # Cannot move forard here

        return delta_x, delta_y

    def get_reward_after_pos_update(self, moved):
        '''
        After updating postion update, get the rewardsbased on whether you moved to correct state
        based on intermediate check and long term indicator 
        '''
        reward = 0.0
        if not moved:
            return 0.0
        if self.phase == 2: # long term
            if self.indicator_is_up():
                if self.xy() == (1,0): # Went to right correctly
                    reward = self.success_r
                elif self.xy() == (-1,0): # Went to left incorrectly
                    reward = self.fail_r
            else:
                if self.xy() == (1,0): # Went to right incorrectly
                    reward = self.fail_r
                elif self.xy() == (-1,0): # Went to left correctly
                    reward = self.success_r
        elif self.phase == 1: # intermediate check
            if self.cur_room_type() == 1:
                if self.xy() == (1,1): # Went to right correctly
                    reward = self.check_success_r
                elif self.xy() == (-1,1): # Went to left incorrectly
                    reward = self.check_fail_r
            else:
                assert self.cur_room_type() == 0
                if self.xy() == (1,1): # Went to right incorrectly
                    reward = self.check_fail_r
                elif self.xy() == (-1,1): # Went to left correctly
                    reward = self.check_success_r
        return reward

    def step(self, action):
        if self.timeout is not None and self.step_num >= self.timeout:
            return self.get_obs(), 0.0, True, {}

        reward = 0.0

        delta_x, delta_y = self.get_delta_x_delta_y(action)

        # Deal with rewards for progress (before pos update)
        moved = (delta_x != 0 or delta_y != 0)
        if moved and \
           not self.phase == 2 and \
           not (self.phase == 1 and self.xy() == (0,1)):
            reward += self.reward_per_progress

        # Update agent pos
        self.agent_x += delta_x
        self.agent_y += delta_y

        # Take care of room/phase transitions and done (given pos update)
        done = False
        if self.phase == 0:
            if self.agent_x >= len(self.x_2_icolor):
                self.phase = 1
                self.agent_x, self.agent_y = 0, 0
        elif self.phase == 1:
            if self.agent_y >= 4:
                self.agent_x, self.agent_y = 0, 0
                self.room_num += 1
                if self.room_num >= self.num_rooms:
                    self.phase = 2
        else:
            if self.agent_x != 0:
                done = True # DONE
            assert self.phase == 2, self.phase

        # Deal with rewards for intermediate checks and long term indicator
        reward += self.get_reward_after_pos_update(moved)

        if not done:
            assert self.xy() in self.phase_2_valid_xy[self.phase], self.xy()

        self.step_num += 1

        return self.get_obs(), reward, done, {}

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', mode
        return self.last_unnormed_obs


def env_loop(env, config, args):
    # Run environment
    for i in range(args.episodes):
        print("\n\nStarting New Episode " + str(i) + "\n\n")

        obs = env.reset()
        obs = env.reset_task()
        obs = env.get_obs(reshape=False)
        write_obs(obs)
        if not args.no_sleep:
            time.sleep(1)

        done = False
        step = 0
        tot_r = 0
        while not done:
            print("\nOn step:", step)
            print("About to step. What action?")

            if args.random: # rand agent
                a = env.action_space.sample()
                print("randomly selected:", a)
            elif args.solve: # opt agent
                len_phase0 = 2 if config["multi_step_indicator"] else config["num_single_step_repeats"]
                if step < len_phase0: # phase 0
                    a = 2 # Move right at beginning
                elif step >= (len_phase0 + config["num_rooms"]*6): # phase 2 (at the end)
                    correct_a = 2 if env.indicator_is_up() else 0
                    inncorrect_a = 0 if env.indicator_is_up() else 2
                    if args.subopt:
                        correct_incorrect_probs = [0.75, 0.25] if config["multi_step_indicator"] else [0.5, 0.5]
                        a = np.random.choice([correct_a, inncorrect_a], p=correct_incorrect_probs)
                    else: # args.end_strat == "opt"
                        a = correct_a
                else: # phase 1
                    step_in_room = (step-len_phase0)%6
                    if step_in_room == 1:
                        a = 0 if env.cur_room_type()==0 else 2
                    elif step_in_room in [0, 2, 3, 5]:
                        a = 1
                    else:
                        assert step_in_room == 4, step_in_room
                        a = 2 if env.cur_room_type()==0 else 0
                print("optimally selected:", a)
            else: # prompt user for input
                # For convienience, map ` to 0
                print("Forward=1, Right=2, Left=0 or `")
                a_str = input()
                a_str = "0" if a_str == "`" else a_str
                a = int(a_str)

            obs, reward, done, info = env.step(a)
            obs = env.get_obs(reshape=False)
            print("stepped")

            if not args.no_sleep:
                write_obs(obs)

            print("reward:", reward)
            print("done:", done)
            print("flat obs:", obs.flatten())

            tot_r += reward
            step += 1
            if not args.no_sleep:
                time.sleep(1)
        tot_r = np.round(tot_r, 5)
        print("\nReward for episode: ", tot_r)
        if args.solve:
            # Print expected total r following optimal policy (Q* with gamma=1)
            num_steps = len_phase0 + config["num_rooms"]*6 + 1
            expected_tot_r = 0
            expected_tot_r += (num_steps-config["num_rooms"]-1)*config["reward_per_progress"]
            expected_tot_r += config["num_rooms"]*config["check_success_r"]
            if args.subopt:
                if config["multi_step_indicator"]:
                    expected_tot_r += (3/4)*config["success_r"] + (1/4)*config["fail_r"]
                else:
                    expected_tot_r += 0.5*config["success_r"] + 0.5*config["fail_r"]
            else:
                expected_tot_r += config["success_r"]
            expected_tot_r = np.round(expected_tot_r, 3)
            print("\nExpected Reward for episode: ", expected_tot_r)
            if not args.subopt:
                assert expected_tot_r == tot_r, (expected_tot_r, tot_r)

    env.close()


def main():
    # NOTE: "full" version where minecraft actually launches is slow and having issues skipping frames/rewards
    # additionally, at least with the random agemt, it is having trouble resetting.
    # The other version uses cached images from Minecraft.
    parser = argparse.ArgumentParser(description='Interact with MineMaze env')
    parser.add_argument('--episodes', 
        type=int, 
        default=1, 
        help='Number of episodes to run.')
    parser.add_argument('--random', 
        action="store_true", 
        default=False, 
        help='Whether to let a random agent solve the maze.')
    parser.add_argument('--solve', 
        action="store_true", 
        default=False, 
        help='Whether to let an optimal agent solve the maze. (Note: not compatible with args.full currently)')
    parser.add_argument('--subopt', 
        action="store_true", 
        default=False, 
        help='For solve option, whether act at the end as if your memory is order invariant.')
    parser.add_argument('--downsample', 
        action="store_true", 
        default=False, 
        help='Instead of interacting, use this script to downsample cached 1000x1000 resolution to DOWNSAMPLE_SZxDOWNSAMPLE_SZ.')
    parser.add_argument('--no_sleep', 
        action="store_true", 
        default=False, 
        help="Don't sleep between steps in loop nor write obs to disk")
    args = parser.parse_args()

    # Checks and cleanup based on args
    if args.solve:
        assert not args.random, "Cannot have --solve and --random"

    # Downsample images if args.downsample, then exit
    if args.downsample:
        full_res_dir = "./mine_maze_data/1000_resolution"
        copytree(full_res_dir, LOW_RES_DIR)
        files_to_downsample = glob.glob(os.path.join(LOW_RES_DIR, "**/*.png"), recursive=True)
        for fp in files_to_downsample:
            img = Image.open(fp)
            img = img.resize((DOWNSAMPLE_SZ,DOWNSAMPLE_SZ),Image.ANTIALIAS)
            img.save(fp)
        exit()

    # Create correct env
    config = CACHED_CONFIG
    env = MineMaze(config)

    # Run environment loop. (Either a REPL or automatically solved.)
    env_loop(env, config, args)


if __name__ == '__main__':
    main()