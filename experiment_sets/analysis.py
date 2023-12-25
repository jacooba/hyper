from experiment_sets.models import *


GPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for gpu
CPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for cpu only




# Walkergrad info
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "walker-grad",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "analysis",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 131,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "walker_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "Walker2DRandParams-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(30e6),
        "tbptt_stepsize": None,
        #
        "compute_grad_info": True,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        # RL2:
        RNN_HN,
        RNN_S,
        RNN,
        RNN_HN_Kaiming,
        ]
    })


# some checks on experiments above
check_exps(CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS)


