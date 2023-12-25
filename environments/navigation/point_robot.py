import numpy as np
import matplotlib.pyplot as plt
import torch
from gym import Env
from gym import spaces
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100):
        self.reset_task()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(2,))
        self._max_episode_steps = max_episode_steps

    def sample_task(self):
        goal = np.array([np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)])
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state.flat
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task()}
        return ob, reward, done, info

    

class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''

    def __init__(self, goal_radius=0.2, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.goal_radius = goal_radius
        self.reset_task()

    def sample_task(self):
        radius = 1.0
        angle = np.random.uniform(0, np.pi)
        xs = radius * np.cos(angle)
        ys = radius * np.sin(angle)
        return np.array([xs, ys])

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        # return ob, reward, done, d
        return ob, sparse_reward, done, d

class SemicircleEnv(Env):
    '''
     - tasks sampled from unit half-circle. Modified to be more like Humplik et. al., 2019
    '''
    def __init__(self, goal_radius=0.04, radius=.2, max_episode_steps=10):
        self.radius = radius
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = spaces.Box(low=np.array([-0.01,-2*np.pi/100]), high=np.array([0.01,2*np.pi/100]))
        self._max_episode_steps = max_episode_steps
        self.goal_radius = goal_radius
        self.task_dim = 3
        self.step_count = 0
        self.reset()
        self.reset_task()

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)

    def reset(self):
        self.step_count = 0
        return self.reset_pos()

    def _get_obs(self):
        return np.copy(self._state)

    def sample_task(self):
        angle = np.random.uniform(0, np.pi)
        xs = self.radius * np.cos(angle)
        ys = self.radius * np.sin(angle)
        return np.array([xs, ys, angle])

    def reset_pos(self):
        self._state = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
        return self._get_obs()

    def step(self, action):
        for _ in range(10):
            # update state with action
            x, y, angle = self._state.flat
            vel, angle_vel = action
            angle += angle_vel
            x += vel * np.cos(angle)
            y += vel * np.sin(angle)
            self._state = np.array([x, y, angle])
            # clamp to semicircle in state
            x, y, angle = self._state.flat
            x = min(x, self.radius)
            x = max(x, -self.radius)
            y = min(y, abs((self.radius**2- x**2))**.5)
            y = max(y, -abs((self.radius**2 - x**2))**.5)
            self._state = np.array([x, y, angle%(2*np.pi)])
            # make reward sparse and reset to origin
            distance_to_goal = ((self._goal[0] - x)**2 + (self._goal[1] - y)**2) ** 0.5
            if distance_to_goal <= self.goal_radius:
                reward = 1
                self.reset_pos()
                break
            else:
                reward = 0
        # d.update({'reward': reward})
        # return ob, reward, done, d
        done = self.step_count >= self._max_episode_steps
        return self._get_obs(), reward, done, {'task': self.get_task()}

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
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
        pos_rewarded = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env._state.copy()

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
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
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
                pos[episode_idx].append(unwrapped_env._state.copy())
                if rew > 0:
                    pos_rewarded[episode_idx].append(pos[episode_idx][-2]) # add previous position

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
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = unwrapped_env._state.copy()
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
        # print(pos)
        plt.figure(figsize=(5, 4 * num_episodes))
        min_dim = -.3
        max_dim = .3
        span = max_dim - min_dim

        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1)

            # draw goal with radius
            curr_task = env.get_task()
            goal = plt.Circle(curr_task, 0.04, color='r')
            plt.gca().add_patch(goal)

            # draw circle
            x = np.linspace(-.2, .2)
            y = np.abs((.04 - x**2))**.5
            plt.plot(x, y, 'b')

            # draw agent path and points
            x = list(map(lambda p: p[0], pos[i]))
            y = list(map(lambda p: p[1], pos[i]))
            plt.scatter(x, y, 8, 'g') # points
            plt.plot(x, y, 'g', linestyle='--') # path

            # draw points rewarded in red to highlight
            x = list(map(lambda p: p[0], pos_rewarded[i]))
            y = list(map(lambda p: p[1], pos_rewarded[i]))
            plt.scatter(x, y, 12, 'r') # points

            # draw agent orientation
            start_angle = pos[i][0][2]
            plt.plot([0,0.02*np.cos(start_angle)], [0,0.02*np.sin(start_angle)], 'r', linestyle="-")

            plt.title('task: {}'.format(curr_task), fontsize=15)
            if 'Goal' in args.env_name:
                plt.plot(curr_task[0], curr_task[1], 'rx')

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

        if not return_pos:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos



class PointEnvOracle(PointEnv):
    def __init__(self, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def _get_obs(self):
        return np.concatenate([self._state.flatten(), self._goal])


class SparsePointEnvOracle(SparsePointEnv):
    def __init__(self, max_episode_steps=100):
        super().__init__(max_episode_steps=max_episode_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def _get_obs(self):
        return np.concatenate([self._state.flatten(), self._goal])
