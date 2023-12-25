import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.mujoco.half_cheetah import HalfCheetahEnv
import utils.helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.num_tasks = 2
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return observation, reward, done, infos

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def task_to_id(self, tasks):
        tasks = tasks.to(torch.int64)
        ids = torch.where(tasks > 0, torch.ones_like(tasks), torch.zeros_like(tasks))
        return ids


class HalfCheetahDirSparseEnv(HalfCheetahDirEnv):

    def __init__(self, max_episode_steps=200, sparse_dist=5):
        self.sparse_dist = sparse_dist
        self.belief_dim = 2
        self.initialise_belief()
        super(HalfCheetahDirSparseEnv, self).__init__(max_episode_steps=max_episode_steps)

    def get_belief(self):
        return self.belief

    def initialise_belief(self):
        # the belief is defined as probability of the task being
        # "go left" (index 0) and "go right" (index 1)
        self.belief = np.array([0.5, 0.5])

    def reset_task(self, task=None):
        super().reset_task(task)
        self.initialise_belief()

    def update_belief(self):

        belief_is_prior = (self.belief[0] == 0.5 and self.belief[1] == 0.5)

        # only update belief if we haven't learned anything yet
        if belief_is_prior:
            # if we observe a reward other than zero, this means we made it out of the "no-go" zone
            curr_x_pos = self.get_body_com("torso")[0]
            if np.abs(curr_x_pos) >= self.sparse_dist:
                if self.get_task() == -1:
                    self.belief = np.array([1.0, 0.0])
                elif self.get_task() == 1:
                    self.belief = np.array([0.0, 1.0])
                else:
                    raise ValueError

    def step(self, action):
        observation, reward, done, infos = super().step(action)

        self.update_belief()

        # sparsify reward
        curr_x_pos = self.get_body_com("torso")[0]
        if np.abs(curr_x_pos) < self.sparse_dist:
            reward = infos['reward_ctrl']

        return observation, reward, done, infos

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            intrinsic_reward=None,
                            vae=None,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        # --- roll out policy ---

        # (re)set environment
        env.reset_task()
        state, belief, task = utl.reset_env(env, args)
        state = state.reshape((1, -1)).to(device)
        start_state = state.clone()
        task = task.view(-1) if task is not None else None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[0].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if encoder is not None:
                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_state.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action, _ = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.reshape((1, -1)).to(device)
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        hidden_state, return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                if info[0]['done_mdp'] and not done:
                    start_state = info[0]['start_state']
                    start_state = torch.from_numpy(start_state).float().reshape((1, -1)).to(device)
                    start_pos = unwrapped_env.get_body_com("torso")[0].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # --- visualise movement ---

        # interval in which we want to visualise
        x_lim_min = -self.sparse_dist - 2
        x_lim_max = self.sparse_dist + 2

        # first cut up the trajectory (shortly before crossing the border, shortly after, full trajectory)
        cross_point = np.where(np.abs(pos[0]) > self.sparse_dist)[0]
        if len(cross_point) != 0 and intrinsic_reward is not None:
            cross_point = cross_point[0]
            min_cut = max([0, cross_point - 5])
            max_cut = min([cross_point + 5, len(pos[0]) - 1])
            if max_cut == len(pos[0]) - 1:
                cut_points = [min_cut, -1]
            else:
                cut_points = [-1, min_cut, max_cut]
        else:
            cut_points = [-1]

        if (intrinsic_reward is not None) and (vae is None):
            # then compute reward bonusses for some subset of the trajectory
            reward_bonusses = []
            outputs_prior_hyperstate = []
            outputs_predictor_hyperstate = []
            for cp in cut_points:
                use_posterior = sum(np.abs(pos[0][:cp]) >= self.sparse_dist) > 0
                for i in range(num_episodes):
                    # --- visualise reward bonus in background ---
                    n = 501
                    x_pos = np.linspace(x_lim_min, x_lim_max, n)

                    belief = np.zeros((n, 2)) + 0.5
                    if use_posterior:
                        if self.get_task() == -1:
                            belief[:, 0] = 1
                            belief[:, 1] = 0
                        else:
                            belief[:, 0] = 0
                            belief[:, 1] = 1

                    rew_bonus = intrinsic_reward.reward(
                        state=torch.from_numpy(x_pos[:, np.newaxis]).float().to(device),
                        belief=torch.from_numpy(belief).float().to(device),
                        done=done)
                    reward_bonusses.append(rew_bonus.view(-1).detach().cpu().numpy())

            # normalise the reward bonusses
            min_rew_bonus = min([min(r) for r in reward_bonusses])
            max_rew_bonus = max([max(r) for r in reward_bonusses])

            normalised_reward_bonusses = []
            for i in range(len(reward_bonusses)):
                normalised_reward_bonusses.append(
                    (reward_bonusses[i] - min_rew_bonus) / (max_rew_bonus - min_rew_bonus))

        for k, cp in enumerate(cut_points):

            # --- plot movement ---

            plt.figure(figsize=(7, 4 * num_episodes))

            for i in range(num_episodes):

                plt.subplot(num_episodes, 1, i + 1)

                # (not plotting the last step because this gives weird artefacts)
                plt.plot(pos[i][:cp], range(len(pos[i][:cp])), 'k')

                if (intrinsic_reward is not None) and (vae is None): # for the belief oracle we can show the bonus along the way
                    width = (x_lim_max - x_lim_min) / n * 10
                    for j in range(n):
                        col = (1, 1 - normalised_reward_bonusses[k][j] ** 2, 1 - normalised_reward_bonusses[k][j] ** 2)
                        plt.bar(x_pos[j], self._max_episode_steps, width=width, linewidth=0, facecolor=col)

                if self.get_task() == -1:
                    task = 'go left'
                else:
                    task = 'go right'
                plt.title('Task: {}'.format(task), fontsize=15)
                plt.ylabel('Steps'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('Position', fontsize=15)

                plt.plot([-self.sparse_dist, -self.sparse_dist], [0, self._max_episode_steps], 'k--', alpha=0.5)
                plt.plot([self.sparse_dist, self.sparse_dist], [0, self._max_episode_steps], 'k--', alpha=0.5)

                plt.xlim([x_lim_min, x_lim_max])
                plt.ylim([0, self._max_episode_steps])

            plt.tight_layout()
            if image_folder is not None:
                plt.savefig('{}/{}_behaviour_{}'.format(image_folder, iter_idx, cp))
                plt.close()
            else:
                plt.show()

            # ------- visualise reward bonus over time when we run VariBAD -------

            if vae is not None:

                rew_bonus, intrinsic_rew_state, \
                intrinsic_rew_belief, intrinsic_rew_hyperstate, \
                intrinsic_rew_vae_loss = intrinsic_reward.reward(state=torch.cat(episode_next_obs, dim=1),
                                                                 belief=torch.cat((episode_latent_means[0][1:], episode_latent_logvars[0][1:]), dim=-1),
                                                                 action=episode_actions[0],
                                                                 done=done,
                                                                 return_individual=True,
                                                                 vae=vae,
                                                                 latent_mean=[episode_latent_means[0][1:]],
                                                                 latent_logvar=[episode_latent_logvars[0][1:]],
                                                                 batch_prev_obs=torch.stack(episode_prev_obs, dim=0),
                                                                 batch_next_obs=torch.stack(episode_next_obs, dim=0),
                                                                 batch_actions=torch.stack(episode_actions, dim=0),
                                                                 batch_rewards=torch.stack(episode_rewards, dim=0),
                                                                 )

                for name, my_rew in [['state_bonus', intrinsic_rew_state],
                                     ['belief_bonus', intrinsic_rew_belief],
                                     ['hyperstate_bonus', intrinsic_rew_hyperstate],
                                     ['vae_loss_bonus', intrinsic_rew_vae_loss]]:

                    if isinstance(my_rew, int):
                        continue

                    plt.plot(my_rew.detach().cpu().view(-1))
                    plt.xlabel('Steps')
                    plt.ylabel('Bonys')

                    plt.legend()
                    plt.tight_layout()
                    if image_folder is not None:
                        plt.savefig('{}/{}_rew_bonus_{}'.format(image_folder, iter_idx, name))
                        plt.close()
                    else:
                        plt.show()

        if not return_pos:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos

