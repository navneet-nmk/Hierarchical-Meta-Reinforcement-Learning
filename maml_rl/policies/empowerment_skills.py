"""
Run EmpowermentSkills on HalfCheetahDir Environment.
"""

import numpy as np
import torch
import rlkit.rlkit.torch.pytorch_util as ptu
import torch.optim as optim
from rlkit.rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.rlkit.samplers.data_collector import MdpPathCollector
from rlkit.rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.rlkit.torch.networks import FlattenMlp
from rlkit.rlkit.envs.wrappers import NormalizedBoxEnv
from gym.envs.mujoco import HalfCheetahEnv
from rlkit.rlkit.launchers.launcher_util import setup_logger
from maml_rl.policies import emp_skills_trainer, categorical_mlp

print("Using the torch version : ", torch.__version__)

EPS = 1E-6


def experiment(variant):
    eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    skills_dim = variant['skills_dim']

    # Define the networks

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skills_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skills_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skills_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skills_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + skills_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    higher_level_policy = categorical_mlp.CategoricalMLPPolicy(
        input_size=obs_dim,
        output_size=skills_dim,
        hidden_sizes=(M, M),
    )

    value_function = FlattenMlp(
        hidden_sizes=[M, M],
        input_size=obs_dim,
        output_size=1,
    )

    discriminator_function = FlattenMlp(
        hidden_sizes=[M, M],
        input_size=obs_dim,
        output_size=skills_dim
    )

    target_vf = FlattenMlp(
        hidden_sizes=[M, M],
        input_size=obs_dim,
        output_size=1,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        higher_level_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        higher_level_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    trainer = emp_skills_trainer.EmpowermentSkillsTrainer(
        env=eval_env,
        higher_level_policy=higher_level_policy,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_vf=target_vf,
        value_function=value_function,
        discriminator=discriminator_function,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        skills_dim=32,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)