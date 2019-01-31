import numpy as np
import gym
from gym import spaces
from gym.spaces.box import Box
from maml_rl.serializable import Serializable
from maml_rl.proxy_env import ProxyEnv


class NormalizedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        ProxyEnv.__init__(self, env)
        Serializable.quick_init(self, locals())
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.flat_dim)
        self._obs_var = np.ones(env.observation_space.flat_dim)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def _update_obs_estimate(self, obs):
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + self._reward_alpha * np.square(reward -
                                                                                                        self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self, reset_args=None):
        ret = self._wrapped_env.reset(reset_args=reset_args)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["_obs_mean"] = self._obs_mean
        d["_obs_var"] = self._obs_var
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_var = d["_obs_var"]

    @property
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            return spaces.Box(-1 * ub, ub)
        return self._wrapped_env.action_space

    def step(self, action):
        if isinstance(self._wrapped_env.action_space, Box):
            # rescale the action
            lb, ub = self._wrapped_env.action_space.bounds
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        info = {}
        return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    # def log_diagnostics(self, paths):
    #     print "Obs mean:", self._obs_mean
    #     print "Obs std:", np.sqrt(self._obs_var)
    #     print "Reward mean:", self._reward_mean
    #     print "Reward std:", np.sqrt(self._reward_var)

normalize = NormalizedEnv

class NormalizedActionWrapper(gym.ActionWrapper):
    """Environment wrapper to normalize the action space to [-1, 1]. This 
    wrapper is adapted from rllab's [1] wrapper `NormalizedEnv`
    https://github.com/rll/rllab/blob/b3a28992eca103cab3cb58363dd7a4bb07f250a0/rllab/envs/normalized_env.py

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, env):
        super(NormalizedActionWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
            shape=self.env.action_space.shape)

    def action(self, action):
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        # Map the normalized action to original action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = lb + 0.5 * (action + 1.0) * (ub - lb)
        return action

    def reverse_action(self, action):
        # Map the original action to normalized action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = 2.0 * (action - lb) / (ub - lb) - 1.0
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action

class NormalizedObservationWrapper(gym.ObservationWrapper):
    """Environment wrapper to normalize the observations with a running mean 
    and standard deviation. This wrapper is adapted from rllab's [1] 
    wrapper `NormalizedEnv`
    https://github.com/rll/rllab/blob/b3a28992eca103cab3cb58363dd7a4bb07f250a0/rllab/envs/normalized_env.py

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, env, alpha=1e-3, epsilon=1e-8):
        super(NormalizedObservationWrapper, self).__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        shape = self.observation_space.shape
        dtype = self.observation_space.dtype or np.float32
        self._mean = np.zeros(shape, dtype=dtype)
        self._var = np.ones(shape, dtype=dtype)

    def observation(self, observation):
        self._mean = (1.0 - self.alpha) * self._mean + self.alpha * observation
        self._var = (1.0 - self.alpha) * self._var + self.alpha * np.square(observation - self._mean)
        return (observation - self._mean) / (np.sqrt(self._var) + self.epsilon)

class NormalizedRewardWrapper(gym.RewardWrapper):
    """Environment wrapper to normalize the rewards with a running mean 
    and standard deviation. This wrapper is adapted from rllab's [1] 
    wrapper `NormalizedEnv`
    https://github.com/rll/rllab/blob/b3a28992eca103cab3cb58363dd7a4bb07f250a0/rllab/envs/normalized_env.py

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, env, alpha=1e-3, epsilon=1e-8):
        super(NormalizedRewardWrapper, self).__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self._mean = 0.0
        self._var = 1.0

    def reward(self, reward):
        self._mean = (1.0 - self.alpha) * self._mean + self.alpha * reward
        self._var = (1.0 - self.alpha) * self._var + self.alpha * np.square(reward - self._mean)
        return (reward - self._mean) / (np.sqrt(self._var) + self.epsilon)


def normalize_env(env):
    n_env = NormalizedActionWrapper(env)
    n_env = NormalizedObservationWrapper(n_env)
    n_env = NormalizedRewardWrapper(n_env)
    return n_env
