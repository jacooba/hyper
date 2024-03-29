import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.mujoco.ant import AntEnv
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")


class AntGoalSparseEnv(AntEnv):
    def __init__(self, max_episode_steps, level, *args, **kwargs):
        self.init_serialization(locals())
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        if level == 0:
            self.goal_radius = 3.0
        elif level == 1:
            self.goal_radius = 2.0
        elif level == 2:
            self.goal_radius = 1.5
        elif level == 3:
            self.goal_radius = 1.0
        elif level == 4:
            self.goal_radius = 0.75
        elif level == 5:
            self.goal_radius = 0.5
        elif level == 6:
            self.goal_radius = 0.3
        super(AntGoalSparseEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        dist = np.linalg.norm(self.goal_pos - xposafter[:2], ord=2)
        # Only give reward if within a goal radius
        if dist <= self.goal_radius:
            success = True
            goal_reward = -dist
        # otherwise give highest possible penalty
        else:
            success = False
            goal_reward = -self.goal_radius

        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        ctrl_cost = .01 * np.square(action).sum()
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()
        info = {
            'goal_forward': goal_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': survive_reward,
            'task': self.get_task(),
            'success': success,
        }
        return ob, reward, done, info

    def sample_tasks(self, num_tasks):
        # using python builtin random for thread-safeness (numpy's random isn't thread safe)
        a = np.array([random.uniform(0, 1) for _ in range(num_tasks)]) * 2 * np.pi
        r = 3 * np.array([random.uniform(0, 1) for _ in range(num_tasks)]) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return self.goal_pos

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso")
        ])

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        # Note: First just setting it up so we can see where the ant is going. Still need to put in the goal

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
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        # --- roll out policy ---

        # (re)set environment
        env.reset_task()
        state, belief, task = utl.reset_env(env, args)
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                else:
                    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

            if encoder is not None:
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    prev_obs = start_obs_raw.clone()
                else:
                    prev_obs = state.clone()
                episode_prev_obs[episode_idx].append(prev_obs)

                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action, _ = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task,
                                          deterministic=True)

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.float().reshape((1, -1)).to(device)
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device), prev_obs,
                        hidden_state, return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot the movement of the ant
        plt.figure(figsize=(5, 4 * num_episodes))
        min_dim = -3.5
        max_dim = 3.5
        span = max_dim - min_dim

        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1, aspect='equal')

            x = list(map(lambda p: p[0], pos[i]))
            y = list(map(lambda p: p[1], pos[i]))

            # starting spot
            plt.plot(x[0], y[0], 'bo')
            # movement
            plt.scatter(x, y, 1, 'g')
            # title
            curr_task = env.get_task()
            if 'Dir' in args.env_name:
                plt.title('task: {}'.format(curr_task), fontsize=15)
            # goal
            if 'Goal' in args.env_name:
                plt.plot(curr_task[0], curr_task[1], 'rx')

            # radius where we get reward
            circle1 = plt.Circle(self.goal_pos, self.goal_radius, color='c', alpha=0.3, edgecolor='none')
            plt.gca().add_artist(circle1)
            # plot agent's trajectory
            # plt.scatter(x, y, 1, 'r')

            # label the axes
            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            if i == num_episodes - 1:
                plt.xlabel('x-position', fontsize=15)
                plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            plt.xlim(min_dim - 0.05 * span, max_dim + 0.05 * span)
            plt.ylim(min_dim - 0.05 * span, max_dim + 0.05 * span)

        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

        plt_rew = [episode_rewards[i][:episode_lengths[i]] for i in range(len(episode_rewards))]
        plt.plot(torch.cat(plt_rew).view(-1).cpu().numpy())
        plt.xlabel('env step')
        plt.ylabel('reward per step')
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_rewards'.format(image_folder, iter_idx))
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
