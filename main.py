"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import os

import torch

# get configs
from config import default_conf, args_sparse_pointrobot_varibad, args_semicircle_varibad
from config.metaworld import \
    args_ml1_reach_varibad, args_ml1_reach_hyperx, \
    args_ml1_push_varibad, args_ml1_push_hyperx, \
    args_ml1_pickplace_varibad, args_ml1_pickplace_hyperx, \
    args_ml10_varibad
from config.gridworld import \
    args_grid_oracle, args_grid_belief_oracle, args_grid_rl2, args_grid_varibad, args_TLN_varibad, \
    args_TLNP1_varibad, args_TLSP1_varibad, hall, args_MCLSO_varibad, args_MCLS_varibad, args_MCLSH_varibad, \
    args_MCLSN_varibad, args_plan
from config.mujoco import \
    args_cheetah_dir_oracle, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_oracle, args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_vel_avg, \
    args_ant_dir_oracle, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_oracle, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_walker_oracle, args_walker_avg, args_walker_rl2, args_walker_varibad, args_hop_walk, args_hop_walk_nonpara, args_hop, \
    args_sparse_ant_goal_varibad, args_sparse_cheetah_dir_varibad
from metalearner import MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='ant_dir_rl2')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable actions for debugging that may be slow')
    parser.add_argument('--hop_test', action='store_true', default=False,
                        help='runs a test function in hop_walk env instead')
    args, rest_args = parser.parse_known_args()
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    if args.hop_test:
        from environments.mujoco.rand_param_envs.hop_walk import test
        test()
        exit()
    env = args.env_type

    # --- GridWorld ---

    if env == 'gridworld_oracle':
        args, unkown_args = args_grid_oracle.get_args(rest_args)
    elif env == 'gridworld_belief_oracle':
        args, unkown_args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
    elif env == 'GridNavi-dense':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'GridNavi-dense-v0'
    elif env == 'GridNavi-show_start':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'GridNavi-show_start-v0'
    elif env == 'grid7-15':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-15-v0'
    elif env == 'grid7-21':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-21-v0'
    elif env == 'grid7-15-mid':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-15-mid-v0'
    elif env == 'grid8-15-mid':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid8-15-mid-v0'
    elif env == 'grid7-15-mid-ring':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-15-mid-ring-v0'
    elif  env == 'grid7-15-mid-ring-newr':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-15-mid-ring-newr-v0'
    elif env == 'grid25-50-mid-ring-newr':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid25-50-mid-ring-newr-v0'
    elif env == 'grid16-Hall1-H20-rshape':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid16-Hall1-H20-rshape-v0'
    elif env == 'grid60-Hall1-H80-rshape':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid60-Hall1-H80-rshape-v0'
    elif env == 'grid7-20-newr-rands':
        args, unkown_args = args_grid_varibad.get_args(rest_args)
        args.env_name = 'Grid7-20-newr-rands-v0'
    elif env == 'gridworld_rl2':
        args, unkown_args = args_grid_rl2.get_args(rest_args)
    elif env == 'T-LN':
        args, unkown_args = args_TLN_varibad.get_args(rest_args)
    elif env == 'T-LN-P1':
        args, unkown_args = args_TLNP1_varibad.get_args(rest_args)
    elif env == 'T-LS-P1':
        args, unkown_args = args_TLSP1_varibad.get_args(rest_args)
    elif env == 'T-LN-P1-A50':
        args, unkown_args = args_TLNP1_varibad.get_args(rest_args)
        args.env_name = 'T-LN-P1-A50-v0'
    elif env == "T-LN-P1-LDp7p5":
        args, unkown_args = args_TLNP1_varibad.get_args(rest_args)
        args.env_name = "T-LN-P1-LDp7p5-v0"
    elif env == 'hall1':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H80-rshape-v0'
    elif env == 'hall200r':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H200-rshape-v0'
    elif env == 'hall200':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H200-v0'
    elif env == 'hall200ro':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H200-obs-rshape-v0'
    elif env == 'hall200o':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H200-obs-v0'
    elif env == 'hall200ro5':
        args, unkown_args = hall.get_args(rest_args)
        args.env_name = 'Hall-L60H200-obs-rshape5x-v0'
    elif env == 'MCLSO_varibad':
        args, unkown_args = args_MCLSO_varibad.get_args(rest_args)
    elif env == 'MCLS_varibad':
        args, unkown_args = args_MCLS_varibad.get_args(rest_args)
    elif env == 'MCLSH_varibad':
        args, unkown_args = args_MCLSH_varibad.get_args(rest_args)
    elif env == 'MCLSN_varibad':
        args, unkown_args = args_MCLSN_varibad.get_args(rest_args)
    elif env == 'plan3x3':
        args, unkown_args = args_plan.get_args(rest_args)
        args.env_name = 'PlanningGame-3x3-v0'
    elif env == 'plan3x3justgoal':
        args, unkown_args = args_plan.get_args(rest_args)
        args.env_name = 'PlanningGame-3x3-justgoal-v0'

    # --- Point Robot ---
    elif env == 'sparse_point_varibad':
        args, unkown_args = args_sparse_pointrobot_varibad.get_args(rest_args)
    elif env == 'semicircle_varibad':
        args, unkown_args = args_semicircle_varibad.get_args(rest_args)
        args.env_name = 'Semicircle-v0'


    # --- MUJOCO ---

    # - AntDir -
    elif env == 'ant_dir_oracle':
        args, unkown_args = args_ant_dir_oracle.get_args(rest_args)
    elif env == 'ant_dir_rl2':
        args, unkown_args = args_ant_dir_rl2.get_args(rest_args)
    elif env == 'ant_dir_varibad':
        args, unkown_args = args_ant_dir_varibad.get_args(rest_args)
    #
    # - AntGoal -
    elif env == 'ant_goal_oracle':
        args, unkown_args = args_ant_goal_oracle.get_args(rest_args)
    elif env == 'ant_goal_varibad':
        args, unkown_args = args_ant_goal_varibad.get_args(rest_args)
    elif env == 'ant_goal_rl2':
        args, unkown_args = args_ant_goal_rl2.get_args(rest_args)
    elif env == 'sparse_ant_goal_varibad':
        args, unkown_args = args_sparse_ant_goal_varibad.get_args(rest_args)
    #
    # - CheetahDir -
    elif env == 'cheetah_dir_oracle':
        args, unkown_args = args_cheetah_dir_oracle.get_args(rest_args)
    elif env == 'cheetah_dir_rl2':
        args, unkown_args = args_cheetah_dir_rl2.get_args(rest_args)
    elif env == 'cheetah_dir_varibad':
        args, unkown_args = args_cheetah_dir_varibad.get_args(rest_args)
    elif env == 'sparse_cheetah_dir_varibad':
        args, unkown_args = args_sparse_cheetah_dir_varibad.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'cheetah_vel_oracle':
        args, unkown_args = args_cheetah_vel_oracle.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args, unkown_args = args_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'cheetah_vel_varibad':
        args, unkown_args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_avg':
        args, unkown_args = args_cheetah_vel_avg.get_args(rest_args)
    #
    # - Walker -
    elif env == 'walker_oracle':
        args, unkown_args = args_walker_oracle.get_args(rest_args)
    elif env == 'walker_avg':
        args, unkown_args = args_walker_avg.get_args(rest_args)
    elif env == 'walker_rl2':
        args, unkown_args = args_walker_rl2.get_args(rest_args)
    elif env == 'walker_varibad':
        args, unkown_args = args_walker_varibad.get_args(rest_args)
    #
    elif env == 'hop_walk':
        args, unkown_args = args_hop_walk.get_args(rest_args)
    elif env == 'hop_walk_nonpara':
        args, unkown_args = args_hop_walk_nonpara.get_args(rest_args)
    elif env == 'hop':
        args, unkown_args = args_hop.get_args(rest_args)
    # --- Meta-World: ML1 ---
    # - reach
    elif env == 'metaworld_ml1_reach_varibad':
        args, unkown_args = args_ml1_reach_varibad.get_args(rest_args)
    elif env == 'metaworld_ml1_reach_hyperx':
        args, unkown_args = args_ml1_reach_hyperx.get_args(rest_args)
    # push
    elif env == 'metaworld_ml1_push_varibad':
        args, unkown_args = args_ml1_push_varibad.get_args(rest_args)
    elif env == 'metaworld_ml1_push_hyperx':
        args, unkown_args = args_ml1_push_hyperx.get_args(rest_args)
    # pick-place
    elif env == 'metaworld_ml1_pickplace_varibad':
        args, unkown_args = args_ml1_pickplace_varibad.get_args(rest_args)
    elif env == 'metaworld_ml1_pickplace_hyperx':
        args, unkown_args = args_ml1_pickplace_hyperx.get_args(rest_args)

    # --- Meta-World: ML10 ---
    elif env == 'metaworld_ml10_varibad':
        args, unkown_args = args_ml10_varibad.get_args(rest_args)
    #
    else:
        raise RuntimeError('Config not recognized.')

    # get defaults for unknown args
    default_args = default_conf.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in args:
            setattr(args, default_arg_key, default_arg_value)

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, use num_processes 1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # clean up arguments
    if hasattr(args, 'disable_decoder') and args.disable_decoder and not args.rlloss_through_encoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    if args.rlloss_through_encoder and args.norm_latent_for_policy:
        raise RuntimeError('Cannot backprop RL-loss through encoder and normalise latents. Disable one.')

    if args.pass_latent_to_policy and args.pass_task_to_policy:
        print("Warning: You are passing both the task and latent to the policy. This may not be intended.")

    if 'metaworld' in args.env_name and 'norm_rew_clip_param' not in args:
        raise ValueError("Looks like you're using MetaWorld. "
                         "Please specify --norm_rew_clip_param in your args!")

    if 'PlanningGame' in args.env_name:
        assert args.max_rollouts_per_task == 1, "If you want to use the planning game with > 1 rollout per task,"
                                                "You need to modify the environment to reset graph symbols at end of task"
                                                "and also reset the agent location at the end of each episode."

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()
        with open(os.path.join(learner.logger.full_output_folder, "DONE"), "w+") as done_file:
            done_file.write("\n")


if __name__ == '__main__':
    main()
