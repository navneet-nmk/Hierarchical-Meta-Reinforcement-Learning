import maml_rl.envs
import gym
import numpy as np
import torch
import json

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.policies.empowerment_skills import EmpowermentSkills
from rlkit.rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.rlkit.torch.networks import FlattenMlp

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(7)
seed = 7


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def hierarchical_meta_policy(env, skills_dim, sampler, output_size, net_size):
    higher_policy = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        skills_dim,
        hidden_sizes=(args.hidden_size,) * args.num_layers)

    observation_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))

    hidden_size = net_size
    output_size = output_size
    skills_dim = skills_dim

    # Define the networks

    q_value_function_1 = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim + action_dim + skills_dim,
        output_size=output_size)

    q_value_function_2 = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim + action_dim + skills_dim,
        output_size=output_size)

    value_function = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim,
        output_size=output_size
    )

    discriminator_function = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim,
        output_size=skills_dim
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[hidden_size, hidden_size],
        obs_dim=observation_dim + skills_dim,
        action_dim=action_dim
    )

    # Define the empowerment skills algorithm
    algorithm = EmpowermentSkills(env=env,
                                  policy=policy,
                                  discriminator=discriminator_function,
                                  q_value_function_1=q_value_function_1,
                                  q_value_function_2=q_value_function_2,
                                  value_function=value_function)

    lower_policy = algorithm
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape))
    )

    return higher_policy, lower_policy, baseline


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'Pusher'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
            total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
            total_rewards([ep.rewards for _, ep in episodes]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy, f)

    with open(os.path.join(save_folder, 'baseline.pt'), 'wb') as f:
        torch.save(baseline, f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='Pusher',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.1,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=1000,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml-pusher',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()
    print("Using ",str(mp.cpu_count()-1), " number of workers")

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
