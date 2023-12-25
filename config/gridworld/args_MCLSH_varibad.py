import argparse
from utils.helpers import boolean_argument
from config.gridworld import args_grid_varibad


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # changes from args_grid_varibad:
    parser.add_argument('--env_name', default='MC-LSH-v0', help='environment to train on')
    parser.add_argument('--num_frames', type=int, default=30e6, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=2, help='number of MDP episodes for adaptation')
    parser.add_argument('--input_prev_state', type=boolean_argument, default=True, help='use prev state for rew pred')
    parser.add_argument('--input_action', type=boolean_argument, default=True, help='use prev action for rew pred')
    parser.add_argument('--multihead_for_reward', type=boolean_argument, default=False,
                        help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew_pred_type', type=str, default='deterministic',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')
    parser.add_argument('--size_vae_buffer', type=int, default=10000,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--action_embedding_size', type=int, default=8)
    parser.add_argument('--ppo_num_epochs', type=int, default=2, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=4, help='number of minibatches to split the data')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy_num_steps', type=int, default=800,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--vae_batch_num_trajs', type=int, default=25,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--vae_subsample_elbos', type=int, default=50,
                        help='for how many timesteps to compute the ELBO; None uses all')
    parser.add_argument('--vae_resample_decodes', type=boolean_argument, default=True, help='Instead of vae_subsample_decodes, you can re-sample so they have the same (max) length')


    # take other args from args_grid_varibad
    args, unkown_args = parser.parse_known_args(rest_args)
    default_args, still_unknown_args = args_grid_varibad.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in args:
            setattr(args, default_arg_key, default_arg_value)

    return args, still_unknown_args
