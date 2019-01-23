import numpy as np
import torch

from maml_rl.envs.mujoco.half_cheetah import HalfCheetahDirEnv
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.metalearner import MetaLearner

from tensorboardX import SummaryWriter

ITR = 120

# torch.manual_seed(7)

META_POLICY_PATH = "/Users/navneetmadhukumar/Downloads/pytorch-maml-rl/saves/maml/policy-199.pt"

TEST_TASKS = [
    (5., 5.)
]


def load_meta_learner_params(policy_path, env):
    policy_params = torch.load(policy_path)

    policy = NormalMLPPolicy(
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)),
        hidden_sizes=(100, 100))  # We should actually get this from config
    policy.load_state_dict(policy_params)

    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))

    return policy, baseline


def evaluate(env, task, policy, max_path_length=100):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    for _ in range(max_path_length):
        obs_tensor = torch.from_numpy(obs).to(device='cpu').type(torch.FloatTensor)
        action_tensor = policy(obs_tensor, params=None).sample()
        action = action_tensor.cpu().numpy()
        obs, rew, done, _ = env.step(action)
        cum_reward += rew
        t += 1
        if done:
            break

    print("========EVAL RESULTS=======")
    print("Return: {}, Timesteps:{}".format(cum_reward, t))
    print("===========================")

    return cum_reward


def main():
    env = HalfCheetahDirEnv()
    policy, baseline = load_meta_learner_params(META_POLICY_PATH, env)
    sampler = BatchSampler(env_name='HalfCheetahDir-v1',
                           batch_size=20)
    learner = MetaLearner(sampler, policy, baseline)
    writer = SummaryWriter()

    cum_reward = 0

    for i, task in enumerate(TEST_TASKS):
        env.reset_task(task)

        # Sample a batch of transitions
        sampler.reset_task(task)
        episodes = sampler.sample(policy)
        new_params = learner.adapt(episodes)
        policy.load_state_dict(new_params)
        cum_reward = evaluate(env, task, policy)

        writer.add_scalar('data/cumm_reward', cum_reward, i)


if __name__ == '__main__':
    main()