import numpy as np

from environments.metaworld.metaworld.policies.action import Action
from environments.metaworld.metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerSweepIntoV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'cube_pos': obs[3:6],
            'unused_info': obs[6:9],
            'goal_pos': obs[9:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos'] + np.array([-0.005, .0, .01])
        pos_goal = o_d['goal_pos']

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04:
            return pos_cube + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_cube[2]) > 0.04:
            return pos_cube
        else:
            return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos']

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_cube[2]) > 0.15:
            return -1.
        else:
            return .7
