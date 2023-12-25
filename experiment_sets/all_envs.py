from experiment_sets.models import *


GPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for gpu
CPU_EXPERIMENT_SETS = [] # This will be a list of sets of experiments for cpu only


# Meta-world experiments


# Grid.
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "grid",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "gridworld_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "GridNavi-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "eval_interval": 5,
        "num_frames": int(4e6),
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })


# Grid. Dense Reward
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "grid-dense",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "GridNavi-dense",  # default / shared arguments file (usually env specific)
        "env_name": "GridNavi-dense-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "eval_interval": 5,
        "num_frames": int(4e6), 
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported. 
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })


# Grid. Show Start.
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "grid-show",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "GridNavi-show_start",  # default / shared arguments file (usually env specific)
        "env_name": "GridNavi-show_start-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25, 
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "eval_interval": 5,
        "num_frames": int(4e6), # 8
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported. 
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })


# Ant-Dir
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "ant-dir",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "ant_dir_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "AntDir-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 10, 
        "policy_task_embedding_dim": 10, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(75e6),
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5], 
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })

# Walker
GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "walker",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 131,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "walker_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "Walker2DRandParams-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(75e6),
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5], 
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })

GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "chedir",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "cheetah_dir_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "HalfCheetahDir-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(75e6),
        "tbptt_stepsize": None,
        #
        "ppo_num_epochs": 2, 
        "ppo_num_minibatch": 4,
        "policy_num_steps": 400,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })

GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "chevel",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "cheetah_vel_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "HalfCheetahVel-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(75e6),
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5], 
        "seed": [73, 20, 3],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })

GPU_EXPERIMENT_SETS.append(
    {
    "set_name": "ml10",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 200,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "metaworld_ml10_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "metaworld_ml10",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "num_frames": int(1e8),
        "tbptt_stepsize": None,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported. 
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3],
        },
    "joint_search_arguments":{},
    # unique arguments for each experiment / model:
    "experiments": [ # 833 updates also reasonable
        VI_HN,
        RNN_HN,
        ]
    })

GPU_EXPERIMENT_SETS.append( # seems to be best for hyper
    {
    "set_name": "mc_ls",  # Note: name for experiment set. Not used; just for debug and convenience. # aggregate
    "dir_name": "all_envs",       # directory for experiments, can be same across sets so long as env changes
    # mujoco version:
    "mujoco_version": 150,                   # Can leave blank and a default will be assumed
    # shared arguments:
    "shared_arguments":{                  # Note: if the directory stays same, env_name must change:
        "env-type": "MCLS_varibad",  # default / shared arguments file (usually env specific)
        "env_name": "MC-LS-v0",        # env 
        #
        "policy_layers": "32 32", # + head. Must pass as str to be parsed correctly
        "encoder_gru_hidden_size": "256",
        "policy_latent_embedding_dim": 25,
        "policy_task_embedding_dim": 25, ### Needs to be equal to policy_latent_embedding_dim for full_task_chance
        #
        "hypernet_input": "latent",
        "init_hyper_for_policy": True,
        #
        "tbptt_stepsize": None,
        "eval_interval": 10,
        "num_frames": int(60e6),
        #
        "policy_anneal_lr": True,
        },
    # search arguments for hyper-param search:
    "search_arguments":{
        "lr_vae": [0.001],      # Note: search over only lr at a time is supported.
        "lr_policy": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
        "seed": [73, 20, 3, 99,],
        },
    # unique arguments for each experiment / model:
    "experiments": [
        VI_HN,
        RNN_HN,
        ]
    })

# some checks on experiments above
check_exps(CPU_EXPERIMENT_SETS+GPU_EXPERIMENT_SETS)


