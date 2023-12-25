import argparse
from config.mujoco import args_cheetah_dir_varibad

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    parser.add_argument('--env_name', default='Hop-v0', help='environment to train on')

    # take other args from args_cheetah_dir_varibad
    args, unkown_args = parser.parse_known_args(rest_args)
    default_args, still_unknown_args = args_cheetah_dir_varibad.get_args(unkown_args)
    for default_arg_key, default_arg_value in default_args.__dict__.items():
        if not default_arg_key in args:
            setattr(args, default_arg_key, default_arg_value)

    return args, still_unknown_args
