import numpy as np
from gym.spaces import Box

from environments.metaworld_v2.metaworld.envs.asset_path_utils import full_v1_path_for
from environments.metaworld_v2.metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHandlePressEnv(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.05)
        obj_high = (0.1, 0.9, 0.05)
        goal_low = (-0.1, 0.65, 0.0399)
        goal_high = (0.1, 0.75, 0.0401)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.05]),
            'hand_init_pos': np.array( (0, 0.6, 0.2),),
        }
        self.goal = np.array([0, 0.8, 0.14])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_handle_press.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pressDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'goalDist': pressDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pressDist <= 0.04)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self.data.site_xpos[self.model.site_name2id('handleStart')]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos
            button_pos = goal_pos.copy()
            button_pos[1] -= 0.1
            button_pos[2] += 0.09
            self._target_pos = button_pos

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('handle')] = self._target_pos
        self._set_obj_xyz(0)
        self._target_pos = self._get_site_pos('goalPress')
        self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleStart')][-1] - self._target_pos[-1])
        self.target_reward = 1000*self.maxDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        leftFinger = self._get_site_pos('leftEndEffector')
        fingerCOM  =  leftFinger

        pressGoal = self._target_pos[-1]

        pressDist = np.abs(objPos[-1] - pressGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
        else:
            pressRew = 0
        pressRew = max(pressRew, 0)
        reward = -reachDist + pressRew

        return [reward, reachDist, pressDist]
