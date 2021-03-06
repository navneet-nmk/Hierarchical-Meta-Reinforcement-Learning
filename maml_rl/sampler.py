import gym
import torch
import multiprocessing as mp
import time

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.envs.mujoco.pusher import PusherEnv
from maml_rl.envs.normalized_env import normalize_env


def make_env(env_name):
    def _make_env():
        if env_name is 'Pusher':
            return normalize_env(PusherEnv())
        else:
            return gym.make(env_name)
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name,
                 batch_size,
                 max_path_length=100,
                 num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_path_length = max_path_length
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        if env_name is 'Pusher':
            self._env = normalize_env(PusherEnv())
        else:
            self._env = gym.make(env_name)

    def sample_hierarchical(self, l_policy, h_policy, h_params=None,
                            gamma=0.95, device='cpu'):

        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device, hierarchical=True)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        num_samples = 0
        total_samples = 0
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                observations_tensor = observations_tensor.float()
                # We first calculate the latent space actions using the higher level policy
                h_actions_tensor = h_policy(observations_tensor, params=h_params).sample()
                # Now we calculate the lower level actions using the lower level policy
                # Combine the observations
                obs_combined = torch.cat([observations_tensor, h_actions_tensor])
                l_actions_tensor = l_policy(obs_combined)
                actions = l_actions_tensor[0].cpu().numpy()
                h_actions = h_actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.h_append(observations, actions, h_actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
            num_samples += observations.shape[0]
            total_samples += observations.shape[0]

            if num_samples >= self.max_path_length:
                # Reset the episode once you reach the maximum path length
                observations, batch_ids = self.envs.reset()
                num_samples = 0

            if total_samples >= self.batch_size * self.max_path_length:
                break

        return episodes

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        num_samples = 0
        total_samples = 0
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                observations_tensor = observations_tensor.float()
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
            num_samples += observations.shape[0]
            total_samples += observations.shape[0]

            if num_samples >= self.max_path_length:
                # Reset the episode once you reach the maximum path length
                observations, batch_ids = self.envs.reset()
                num_samples = 0

            if total_samples >= self.batch_size*self.max_path_length:
                break

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks