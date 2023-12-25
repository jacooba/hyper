import argparse
from environments.parallel_envs import make_vec_envs
from utils.helpers import boolean_argument

import json
import os
import shutil
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import cv2
from gym.wrappers import Monitor

MC_END_FRAME_REPEAT = 0 # repeat signal in video for MineCraft to make it clear to human
MC_START_FRAME_REPEAT = 0

device = 'cpu' # must use cpu


from utils import helpers as utl

sns.set(style="dark")
sns.set_context("paper")


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def write_video(observations, out_path, fps=3, end_repeat=MC_END_FRAME_REPEAT, start_repeat=MC_START_FRAME_REPEAT):
    # From: https://github.com/jacooba/AMRL-ICLR2020
    observations = [o[:-1].reshape(10,10,3).numpy() for o in observations] # unflatten and remove extra number
    start_imgs = [observations[0] for _ in range(start_repeat)]
    end_imgs = [observations[-1] for _ in range(end_repeat)]
    observations = start_imgs + observations + end_imgs
    shape = observations[0].shape[:-1]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, shape)
    for obs in observations:
        obs = (obs * 128) + 128 # unnormalize
        obs = np.clip(np.round(obs), 0, 256)
        obs = obs.astype(np.uint8)
        obs = np.flip(obs, axis=(0,2))
        writer.write(obs)
    writer.release()


def rollout_varibad(path_to_results, out_path, plot_activations, activation_path, activation_steps, 
    activations_by_step, activation_num_tasks, activation_num_saves, render=False, num_episodes=2):

    path = os.path.join(path_to_results, 'config.json')
    print(path)
    with open(path) as json_data_file:
        args = json.load(json_data_file)
        args = Bunch(args)

    # get policy network
    policy = torch.load(os.path.join(path_to_results, 'models', 'policy.pt'), map_location='cpu').to(device)
    # get encoder
    encoder = torch.load(os.path.join(path_to_results, 'models', 'encoder.pt'), map_location='cpu').to(device)
    # get the normalisation parameters for the environment
    ret_rms = None  # utl.load_obj(os.path.join(path_to_results, 'models'), "env_rew_rms")


    batch_sz = activation_num_tasks if plot_activations else 1

    # create env
    envs = make_vec_envs(args.env_name,
                         seed=args.seed *  random.randint(0, 999),
                         num_processes=batch_sz,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=73,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms,
                         args=args)
    max_episode_steps = envs._max_episode_steps * num_episodes

    for t in range(0,max_episode_steps,max_episode_steps//activation_num_saves):
        activation_steps.append(t)
    activation_steps.append(max_episode_steps-1)

    video = not plot_activations
    if video:
        if "MC" in args.env_name:
            ep_observations = []
        else:
            # hacks to get the video working (not sure why it expects this)
            envs.envs[0].reward_range = [-100, 100]
            envs.enabled = False
            # out_path = '/users/jaceck/MetaMem/video_varibad' # './video_varibad'+str(envs.envs[0].get_task())
            envs.envs[0] = Monitor(envs.envs[0], out_path, force=True)

    # envs.set_task(goal_direction)
    state, belief, task = utl.reset_env(envs, args)
    if render:
        envs.render('human')

    # reset latent state to prior
    latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(batch_sz)
    latent_sample = latent_sample

    all_latent_means = [latent_mean[0]]
    all_latent_logvars = [latent_logvar[0]]
    # May also need:
    # task = envs.get_task()[0] #for some envs, may be: envs.envs[0].get_task()
    tasks = torch.from_numpy(envs.get_task()).to(device)
    # May also need:
    # if args['env_name'] in ['AntDir-v0', 'HalfCheetahDir-v0', 'HalfCheetahDirSparse-v0']:
    #   pos = [[env.get_body_com("torso")[0]], [], []]
    if args.env_name == 'metaworld_ml10' and plot_activations: # just get non-parametric task:
        tasks = tasks[:,:10] 

    returns_per_episode = [0]
    success_per_episode = [False]
    total_r = 0
    for step in range(max_episode_steps):

        ep_observations.append(state[0])
        with torch.no_grad():
            _, action, _ = utl.select_action(args, policy,
                                          state=state.to(device),
                                          deterministic=True,
                                          latent_sample=latent_sample.to(device), latent_mean=latent_mean.to(device),
                                          latent_logvar=latent_logvar.to(device), 
                                          task=tasks if args.pass_task_to_policy else None)

        if plot_activations and step in activation_steps:
            goal_2_activations = {} # save activations by goal, not by env index
            for env_i in range(len(tasks)):
                latent = utl.get_latent_for_policy(args=args, latent_sample=latent_sample.to(device), latent_mean=latent_mean.to(device), latent_logvar=latent_logvar.to(device))
                activations = policy.forward(state[env_i].to(device), latent[env_i].to(device), None, None, return_actor_activations=True)
                g = tuple(tasks[env_i]) #for some envs, may be: envs.envs[0].get_task()
                if g in goal_2_activations:
                    goal_2_activations[g].append(activations)
                else:
                    goal_2_activations[g] = [activations]
            activations_by_step.append(goal_2_activations) # also save that dict by timestep


        # Observe reward and next obs
        (state, belief, task), (rew_raw, rew_normalised), done, info = utl.env_step(envs, action, args)
        if video and "MC" in args.env_name:
            if rew_raw == 4 or rew_raw == -3:
                ep_observations = ep_observations + [state[0] for _ in range(MC_START_FRAME_REPEAT)]
            print(rew_raw)
        # May also need:
        # obs_raw, rew_raw, done, info = envs.step(action[0])
        done = torch.from_numpy(np.array(done, dtype=int)).float().view((-1, 1))

        latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder,
                                                                                      state,
                                                                                      action,
                                                                                      rew_raw,
                                                                                      done,
                                                                                      hidden_state)

        # collect the task means/variance
        all_latent_means.append(latent_mean)
        all_latent_logvars.append(latent_logvar)
        # May also need:
        # if args['env_name'] in ['AntDir-v0', 'HalfCheetahDir-v0', 'HalfCheetahDirSparse-v0']:
        #     pos[k].append(env.get_body_com("torso")[0])

        if batch_sz == 1:
            # count the unnormalised rewards
            returns_per_episode[-1] += rew_raw.item()
            total_r += rew_raw.item()
            if 'success' in info[0]:
                success_per_episode[-1] = success_per_episode[-1] or info[0]['success']

            if info[0]['done_mdp']:
                returns_per_episode.append(0)
                success_per_episode.append(False)

            if render:
                envs.render('human')

            if done:
                break

    envs.close()
    if video and "MC" in args.env_name:
        ep_observations.append(state[0])
        write_video(ep_observations, os.path.join(out_path,"vid.avi"))
        print("Total Reward:", total_r)
        print("Terminal Reward:", rew_raw)

    return returns_per_episode[:-1], success_per_episode[:-1]


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/Users/jake/Desktop/HopWalk_Behavior/HopWalk_Behavior20-hop')
    parser.add_argument('--out_path', default='/users/jaceck/MetaMem/video_varibad')
    parser.add_argument('--plot_activations', action='store_true', default=False, help='whether to plot activations of the policy network instead of video')
    parser.add_argument('--reduce_dims_by_task', action='store_true', default=False, help='whether to do dimension reduction by task if plotting activations')
    parser.add_argument('--tsne', action='store_true', default=False, help='wether to use TSNE instead of PCA for dim reduction of activations')
    parser.add_argument('--activation_path', default='/users/jaceck/MetaMem/activations_varibad')
    parser.add_argument('--activation_num_saves', type=int, default=10, help="number of times to save activations over rollout (plus end)") # [0, 100, 199]
    parser.add_argument('--activation_num_tasks', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=1) # Note this may break things if not equal to the number during training
    args = parser.parse_args()

    if args.plot_activations:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

    plt.figure(figsize=(7, 5))
    methods = ['variBAD', 'RL2', 'PEARL', 'ProMP', 'E-MAML']
    goal_direction = -1

    cols_deep = sns.color_palette("deep", 10)
    cols_bright = sns.color_palette("bright", 10)
    cols_dark = sns.color_palette("dark", 10)
    cols = sns.color_palette("colorblind", 10)
    my_colors = {
        'variBAD': cols_bright[8],
        'E-MAML': cols[4],
        'ProMP': cols[2],
        'Oracle': 'k',
        'RL2': cols[0],
        'PEARL': cols[3],
    }

    path_to_results = args.path

    activations_by_step = [] # saved activations
    activation_steps = [] # timesteps at which activations are saved

    returns = rollout_varibad(path_to_results, args.out_path, 
        args.plot_activations, args.activation_path, activation_steps, 
        activations_by_step, args.activation_num_tasks, args.activation_num_saves, num_episodes=args.num_episodes)
    print(returns[-1])

    if args.plot_activations:
        if os.path.exists(args.activation_path):
            shutil.rmtree(args.activation_path) # clear out dir
        os.mkdir(args.activation_path) # make new dir

        # fill these in:
        task_ids_by_step = [] # e.g. if three steps save [[0,0,0,1,1,0,1], [0,1], [1,1,1,0,1]]
        data_by_step = [] # same as above but with ids replaced by numpy vectors
        all_data = [] # flattened version of data
        task_id_2_all_data = {} # map from task id (not goal) to all data (flat, not by step)
        task_id_2_steps = {} # map from task id (not goal) to the step index for each data pt (flat)
        num_task_ids = 0
        num_steps = 0
        step_for_data_index = []
        for i, goal_2_activations in enumerate(activations_by_step):
            id_2_activations = {} # make a new dict with the goals replaced by a goal ID (used for color)
            task_id = 0
            for _, value in goal_2_activations.items():
                id_2_activations[task_id] = value
                task_id += 1
            num_task_ids = max(num_task_ids, task_id)
            t = activation_steps[i] # get the timestep from the index
            num_steps = max(num_steps, t)
            task_ids = []
            data = []
            for task_id, activation_list in id_2_activations.items():
                # activation_list is now [()]
                for activations in activation_list:
                    task_ids.append(task_id)
                    # vec = activations[-1].detach().numpy() # last hidden later
                    vec = torch.cat(activations).detach().numpy() # all layers
                    data.append(vec) # all layers
                    all_data.append(vec)
                    step_for_data_index.append(t)
                    # save same info but by task, if we want to reduce by tasks:
                    if args.reduce_dims_by_task:
                        if task_id not in task_id_2_all_data:
                            task_id_2_all_data[task_id] = []
                        task_id_2_all_data[task_id].append(vec)
                        if task_id not in task_id_2_steps:
                            task_id_2_steps[task_id] = []
                        task_id_2_steps[task_id].append(t)
                    #
            task_ids_by_step.append(task_ids)
            data_by_step.append(data)

        # reduce data and re-group by step
        small_data_by_step = [[] for _ in range(num_steps+1)]
        if args.reduce_dims_by_task:
            for task_id in range(num_task_ids): # note: relies on same number of tasks_id at ever step
                data_in_task = task_id_2_all_data[task_id]
                dim_reduction = TSNE(n_components=2) if args.tsne else PCA(n_components=2)
                small_data_in_task = dim_reduction.fit_transform(np.array(data_in_task))
                steps_in_task = task_id_2_steps[task_id]
                # add data to small_data_by_step using its step number (in this task)
                for i in range(len(small_data_in_task)):
                    t = steps_in_task[i]
                    small_data_by_step[t].append(small_data_in_task[i])
        else:
            # do PCA/TSNE on all data
            dim_reduction = TSNE(n_components=2) if args.tsne else PCA(n_components=2)
            all_small_data = dim_reduction.fit_transform(np.array(all_data))
            # add data to small_data_by_step using its step number
            for i in range(len(all_small_data)):
                t = step_for_data_index[i]
                small_data_by_step[t].append(all_small_data[i])

        # plot by timestep
        for i in range(len(activations_by_step)):
            t = activation_steps[i] # get the timestep from the index
            task_ids = task_ids_by_step[i]
            data = np.array(data_by_step[i])
            avg_variance = np.var(data, axis=0)
            assert len(avg_variance) == len(data[0])
            avg_variance = np.mean(avg_variance)
            print("avg_variance:", avg_variance)
            small_data = np.array(small_data_by_step[t])
            assert len(small_data[0]==2)
            x = small_data[:,0]
            y = small_data[:,1]
            plt.scatter(x, y, c=task_ids, cmap='viridis_r')
            if not args.tsne:
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
            plt.savefig(os.path.join(args.activation_path, "step="+str(t)))
            plt.cla()
            plt.clf


