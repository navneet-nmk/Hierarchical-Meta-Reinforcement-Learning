import numpy as np
import sys
print(sys.path)
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.twin_sac import TwinSAC


class SACAlgorithm(object):
    def __init__(self, env,
                 observation_dim,
                 action_dim,
                 hidden_size,
                 algo_params,
                 output_size=1):

        self.env = env
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.algo_params = algo_params

        # Define the networks

        self.q_value_function_1 = FlattenMlp(
            hidden_sizes=[self.hidden_size, self.hidden_size],
            input_size=self.observation_dim+self.action_dim,
            output_size=self.output_size)

        self.q_value_function_2 = FlattenMlp(
            hidden_sizes=[self.hidden_size, self.hidden_size],
            input_size=self.observation_dim+self.action_dim,
            output_size=self.output_size)

        self.value_function = FlattenMlp(
            hidden_sizes=[self.hidden_size, self.hidden_size],
            input_size=self.observation_dim,
            output_size=self.output_size
        )

        self.policy = TanhGaussianPolicy(
            hidden_sizes=[self.hidden_size, self.hidden_size],
            obs_dim=observation_dim,
            action_dim=action_dim
        )

        self.algorithm = TwinSAC(
            self.algo_params,
            env=self.env,
            policy=self.policy,
            qf1=self.q_value_function_1,
            qf2=self.q_value_function_2,
            vf=self.value_function,)

        self.algorithm.to(ptu)

    def train(self):
        self.algorithm.train()

    def get_sac_algorithm(self):
        return self.algorithm

    def get_algorithm_components(self):
        return self.q_value_function_1, self.q_value_function_2, self.value_function, self.policy