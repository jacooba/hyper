import argparse
from utils.helpers import boolean_argument
from config.gridworld import args_grid_varibad


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # changes from args_grid_varibad:
    parser.add_argument('--env_name', default='Hall-L60H80-rshape-v0', help='environment to train on') # can change to another hall

    # take other args from args_grid_varibad
    args, unkown_args = parser.parse_known_args(rest_args)
    default_args, still_unknown_args = args_grid_varibad.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in args:
            setattr(args, default_arg_key, default_arg_value)

    return args, still_unknown_args
