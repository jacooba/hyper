from gym.envs.registration import register
from environments.navigation import t_maze

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'Hop-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_hop:HalfCheetahHopEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopWalk-v0',
    entry_point='environments.mujoco.hop_walk:HopWalk',
)

register(
    id='HopWalkNonpara-v0',
    entry_point='environments.mujoco.hop_walk:HopWalk',
    kwargs={'parametric': False,},
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'max_episode_steps': 100},
    max_episode_steps=100,
)


# # Sparse
# # ----------------------------------------
#
register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'max_episode_steps': 100, "goal_radius": 0.2},
    max_episode_steps=100,
)

register(
    'Semicircle-v0',
    entry_point='environments.navigation.point_robot:SemicircleEnv',
    max_episode_steps=10,
)

register(
    'HalfCheetahDirSparseTen-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={
        'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirSparseEnv',
        'sparse_dist': 10.0,
        'max_episode_steps': 400,
    },
    max_episode_steps=400,
)

register(
    'AntGoalSparseFour-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_sparse:AntGoalSparseEnv',
            'max_episode_steps': 200,
            "level": 4},
    max_episode_steps=200
)


#
# # GridWorld
# # ----------------------------------------

register(
    'T-LN-v0', #t_maze.TMAZE_ENV_KEY,
    entry_point='environments.navigation.t_maze:TMaze',
    kwargs={'config':
    {
        "check_up": 1, # Observation for check / noise pointed up
        "check_down": -1,
        "maze_length_upper_bound": None,
        "pos_enc": False,
        "wave_encoding_len": None, #3 # Can be null for no not wave-based encoding
        "intermediate_checks": False,
        "intermediate_indicators": True, # Whether there are intermediate indicators. (Even if no checks, will increase action dimension)
        "reset_intermediate_indicators": True, # whether the intermediate indicators change from episode to episode
        "per_step_reset": True,
        "final_intermediate_indicator": True, # Whether or not there is an intermediate indicator at then end
        "check_reward": 0.1,
        "allow_left": False,
        "force_final_decision": True, # Whether to force the agent to move up or down at end
        "force_right": True, # Whether to force the agent to move right (not at end and not for checks)
        "timeout": 150, # Max steps allowed or null
        "timeout_reward": 0,
        "maze_length": 100,
        "indicator_pos": 0,
        "flipped_indicator_pos": None, #can be null for no duplicate flipped indicator
        "correlated_indicator_pos": None, #can be null for no correlated indicator
        "success_reward": 4.0,
        "fail_reward": -3.0,
        "persistent_reward": 0.0, # Reward given per time step
    }}
)

register(
    'T-LN-P1-v0', #"indicator_pos": 1,
    entry_point='environments.navigation.t_maze:TMaze',
    kwargs={'config':
    {
        "check_up": 1, # Observation for check / noise pointed up
        "check_down": -1,
        "maze_length_upper_bound": None,
        "pos_enc": False,
        "wave_encoding_len": None, #3 # Can be null for no not wave-based encoding
        "intermediate_checks": False,
        "intermediate_indicators": True, # Whether there are intermediate indicators. (Even if no checks, will increase action dimension)
        "reset_intermediate_indicators": True, # whether the intermediate indicators change from episode to episode
        "per_step_reset": True,
        "final_intermediate_indicator": True, # Whether or not there is an intermediate indicator at then end
        "check_reward": 0.1,
        "allow_left": False,
        "force_final_decision": True, # Whether to force the agent to move up or down at end
        "force_right": True, # Whether to force the agent to move right (not at end and not for checks)
        "timeout": 150, # Max steps allowed or null
        "timeout_reward": 0,
        "maze_length": 100,
        "indicator_pos": 1,
        "flipped_indicator_pos": None, #can be null for no duplicate flipped indicator
        "correlated_indicator_pos": None, #can be null for no correlated indicator
        "success_reward": 4.0,
        "fail_reward": -3.0,
        "persistent_reward": 0.0, # Reward given per time step
    }}
)

register(
    'T-LS-P1-v0', #"indicator_pos": 1,
    entry_point='environments.navigation.t_maze:TMaze',
    kwargs={'config':
    {
        "check_up": 1, # Observation for check / noise pointed up
        "check_down": -1,
        "maze_length_upper_bound": None,
        "pos_enc": False,
        "wave_encoding_len": None, # Can be null for no not wave-based encoding
        "intermediate_checks": True,
        "intermediate_indicators": True, # Whether there are intermediate indicators. (Even if no checks, will increase action dimension)
        "reset_intermediate_indicators": True, # whether the intermediate indicators change from episode to episode
        "per_step_reset": True,
        "final_intermediate_indicator": True, # Whether or not there is an intermediate indicator at then end
        "check_reward": 0.1,
        "allow_left": False,
        "force_final_decision": False, # Whether to force the agent to move up or down at end
        "force_right": False, # Whether to force the agent to move right (not at end and not for checks)
        "timeout": 150, # Max steps allowed or null
        "timeout_reward": 0,
        "maze_length": 100,
        "indicator_pos": 1,
        "flipped_indicator_pos": None, # can be null for no duplicate flipped indicator
        "correlated_indicator_pos": None, # can be null for no correlated indicator
        "success_reward": 4.0,
        "fail_reward": -3.0,
        "persistent_reward": 0.0, # Reward given per time step
    }}
)

register(
    'MC-LSO-v0',
    entry_point='environments.navigation.mine_maze:MineMaze',
    kwargs={'config':
    {   
        "num_rooms": 10,
        "multi_step_indicator": True,
        "num_single_step_repeats": 1,
        "success_r": 4,
        "fail_r": -3,
        "check_success_r": 0.1,
        "check_fail_r": 0.0,
        "reward_per_progress": 0.1,
        "timeout": 200,
        "high_res": False,
        "noise": None,
    }}
)

register(
    'MC-LS-v0',
    entry_point='environments.navigation.mine_maze:MineMaze',
    kwargs={'config':
    {   
        "num_rooms": 16,
        "multi_step_indicator": False,
        "num_single_step_repeats": 1,
        "success_r": 4,
        "fail_r": -3,
        "check_success_r": 0.1,
        "check_fail_r": 0.0,
        "reward_per_progress": 0.1,
        "timeout": 200,
        "high_res": False,
        "noise": None,
    }}
)

register(
    'MC-LSN-v0',
    entry_point='environments.navigation.mine_maze:MineMaze',
    kwargs={'config':
    {   
        "num_rooms": 10,
        "multi_step_indicator": False,
        "num_single_step_repeats": 1,
        "success_r": 4,
        "fail_r": -3,
        "check_success_r": 0.1,
        "check_fail_r": 0.0,
        "reward_per_progress": 0.1,
        "timeout": 200,
        "high_res": False,
        "noise": .05,
    }}
)

register(
    'MC-LSH-v0',
    entry_point='environments.navigation.mine_maze:MineMaze',
    kwargs={'hide_signal': True, 'config':
    {   
        "num_rooms": 16,
        "multi_step_indicator": False,
        "num_single_step_repeats": 1,
        "success_r": 4,
        "fail_r": -3,
        "check_success_r": 0.1,
        "check_fail_r": 0.0,
        "reward_per_progress": 0.1,
        "timeout": 200,
        "high_res": False,
        "noise": None,
    }}
)


register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

register(
    'GridNavi-dense-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15, 'distance_reward':True},
)

register(
    'GridNavi-show_start-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15, 'show_goal_at_start':True},
)

register(
    'Grid7-15-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 15},
)

register(
    'Grid7-21-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 21},
)

register(
    'Grid7-15-mid-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 15, 'starting_state':(3.0, 3.0)},
)

register(
    'Grid8-15-mid-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 8, 'num_steps': 15, 'starting_state':(3.0, 3.0)},
)


register(
    'Grid7-15-mid-ring-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 15, 'starting_state':(3.0, 3.0), "ring": True,},
)
register(
    'Grid7-15-mid-ring-newr-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 15, 'starting_state':(3.0, 3.0), "ring": True, "new_state_r":.1,},
)
register(
    'Grid25-50-mid-ring-newr-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 25, 'num_steps': 50, 'starting_state':(12.0, 12.0), "ring": True, "new_state_r":.1,},
)

register(
    'Grid16-Hall1-H20-rshape-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells':16, 'hall_with_door_every':1, 'num_steps':20, "new_state_r":.1,},
)
register(
    'Grid60-Hall1-H80-rshape-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':80, "new_state_r":.1,},
)

register(
    'Grid7-20-newr-rands-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 7, 'num_steps': 20, "new_state_r":.1, 'starting_state':None},
)
# register(
#     'Grid16-Hall3-H20-rshape-stuck-v0',
#     entry_point='environments.navigation.gridworld:GridNavi',
#     kwargs={'num_cells':16, 'hall_with_door_every':3, 'num_steps': 20, "new_state_r":.1, "stuck":True,},
# )
# register(
#     'Grid61-Hall12-H80-rshape-stuck-v0',
#     entry_point='environments.navigation.gridworld:GridNavi',
#     kwargs={'num_cells':61, 'hall_with_door_every':12, 'num_steps': 80, "new_state_r":.1, "stuck":True,},
# )


register(
    'Hall-L60H80-rshape-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':80, "new_state_r":.1,},
)
register(
    'Hall-L60H200-rshape-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':200, "new_state_r":.1,},
)
register(
    'Hall-L60H200-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':200, "new_state_r":None,},
)
register(
    'Hall-L60H200-obs-rshape-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':200, "new_state_r":.1, "obs_checked":True,},
)
register(
    'Hall-L60H200-obs-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':200, "new_state_r":None, "obs_checked":True,},
)
register(
    'Hall-L60H200-obs-rshape5x-v0',
    entry_point='environments.navigation.hall:Hall',
    kwargs={'num_cells':60, 'hall_with_door_every':1, 'num_steps':200, "new_state_r":.1, "obs_checked":True, "success_r":5,},
)
