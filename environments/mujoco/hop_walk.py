import gym
import torch
import numpy as np

import random

# from environments.mujoco.half_cheetah_dir import HalfCheetahDirEnv
# from environments.mujoco.half_cheetah_vel import HalfCheetahVelEnv


class HopWalk():
    def __init__(self, parametric=True):
        # self.envs = [HalfCheetahDirEnv(), HalfCheetahVelEnv()]
        self.envs = [gym.make("HalfCheetahDir-v0"), gym.make("Hop-v0")] # [gym.make("HalfCheetahDir-v0"), gym.make("HalfCheetahHop-v0")]
        self.mode = random.choice(list(range(len(self.envs))))
        self.parametric = parametric
        if not self.parametric:
            self.num_tasks = 2
            self._set_hardcoded_inner_tasks() # this probably isn't needed, but is safer
        self.task_dim = len(self.get_task())

    def task_to_id(self, tasks):
        assert not self.parametric, "Parametric tasks cannot be converted to IDs"
        return (tasks[...,0:1]).to(torch.int64)

    def _set_hardcoded_inner_tasks(self):
        self.envs[0].set_task(1) # run forward
        self.envs[1].set_task(2) # jump high

    def get_task(self):
        return np.concatenate((np.array([self.mode]), self.envs[self.mode].get_task()))

    def set_task(self, task):
        self.mode = int(task[0])
        self.envs[self.mode].set_task(task[1:])
        if not self.parametric:
            self._set_hardcoded_inner_tasks() # this probably isn't needed, but is safer

    def reset_task(self, task):
        if task is not None:
            self.set_task(task)
        # [e.reset_task(task) for e in self.envs]
        self.mode = random.choice(list(range(len(self.envs))))
        if self.parametric:
            self.envs[self.mode].reset_task(task)
        else:
            self._set_hardcoded_inner_tasks()
        return self.get_task()

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """
        # for some reason when running on servers, unwrapped is called on this env when getting task,
        # so "get_task" is not always used. This stops any unwrapping, which should not be necessary ever
        if attr == "unwrapped":
            return self
        attr = self.envs[self.mode].__getattribute__(attr)
        
        if callable(attr):
            def hooked(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result
            return hooked
        else:
            return attr


def test():
    from time import sleep
    env = HopWalk()

    num_ep = 1
    for i in range(num_ep):
        print("ep:", i+1)
        # import pdb; pdb.set_trace()
        # env.unwrapped.debug_prints()
        obs = env.reset()
        # env.unwrapped.debug_prints()
        print("\nobs:", obs)
        env.render()
        done = False
        while not done:
            # action_map = {"a": LEFT, 
            #               "d": RIGHT, 
            #               "w": UP, 
            #               "s": DOWN,
            #               "n": NOOP,}
            # action = None
            # while action not in action_map:
            #     print("Action?")
            #     print("actions allowed: w,a,s,d,n")
            #     action = input()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # env.unwrapped.debug_prints()
            # exit()
            print()
            env.render(mode='human')
            print("obs:", obs)
            print("done:", done)
            print("info:", info)
            print("reward:", reward)
            sleep(.1)
        print("\nEND\n")