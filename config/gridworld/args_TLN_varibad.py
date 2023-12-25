import argparse
from utils.helpers import boolean_argument
from config.gridworld import args_grid_varibad


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # changes from args_grid_varibad:
    parser.add_argument('--env_name', default='T-LN-v0', help='environment to train on')
    parser.add_argument('--input_prev_state', type=boolean_argument, default=True, help='use prev state for rew pred')
    parser.add_argument('--input_action', type=boolean_argument, default=True, help='use prev action for rew pred')
    parser.add_argument('--multihead_for_reward', type=boolean_argument, default=False,
                        help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew_pred_type', type=str, default='deterministic',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')
    parser.add_argument('--action_embedding_size', type=int, default=8)
    parser.add_argument('--policy_num_steps', type=int, default=700,
                        help='number of env steps to do (per process) before updating')

    # take other args from args_grid_varibad
    args, unkown_args = parser.parse_known_args(rest_args)
    default_args, still_unknown_args = args_grid_varibad.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in args:
            setattr(args, default_arg_key, default_arg_value)

    return args, still_unknown_args
