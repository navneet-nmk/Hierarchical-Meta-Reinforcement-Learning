import maml_rl.envs
import gym
import numpy as np
import torch

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):

    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    continuous_actions = (args.env_name in ['AntVelEnv-v1', 'AntDirEnv-v1',
        'HalfCheetahVelEnv-v1', 'HalfCheetahDirEnv-v1', '2DNavigation-v0'])

    save_folder = os.path.join('tmp', args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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

    # Load model
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        policy.load_state_dict(state_dict)

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    args.meta_batch_size = 81
    # velocities = np.linspace(-1., 3., num=args.meta_batch_size)
    # tasks = [{'velocity': velocity} for velocity in velocities]
    tasks = [{'direction': direction} for direction in [-1, 1]]

    for batch in range(args.num_batches):
        episodes = metalearner.sample(tasks)
        train_returns = [ep.rewards.sum(0).cpu().numpy() for ep, _ in episodes]
        valid_returns = [ep.rewards.sum(0).cpu().numpy() for _, ep in episodes]

        with open(os.path.join(save_folder, '{0}.npz'.format(batch)), 'wb') as f:
            np.savez(f, train=train_returns, valid=valid_returns)
        print('Batch {0}'.format(batch))

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='MAML')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--model', type=str,
        help='path to the model checkpoint')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Evaluation
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)