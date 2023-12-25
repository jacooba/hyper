from experiment_sets.models import *


GPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for gpu
CPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for cpu only




GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "film-pickplace2",  # Note: name for experiment set. Not used; just for debug and convenience.
    "dir_name": "init_main_result",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 200,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "metaworld_ml1_pickplace_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "metaworld_ml1",        # env
        "ml1_type": "pick-place",
        #
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 10,
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(5e7),
        "tbptt_stepsize": None
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3, 51, 19, 7, 36, 10, 68, 39],
        },
    "joint_search_arguments":{},
    # unique arguments for each experiment / model:
    "experiments": [
        FiLM_Normc,
        FiLM_Bias_HyperInit,
        VI_HN_noInit,
        VI_HN,
        VI,
        ]
    })




# some checks on experiments above
check_exps(CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS)


