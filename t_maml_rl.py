import torch

from maml_rl.envs.mujoco.half_cheetah import HalfCheetahDirEnv
from maml_rl.sampler import BatchSampler
from maml_rl.metalearner import MetaLearner
from gym.utils import seeding

from tensorboardX import SummaryWriter

ITR = 120

torch.manual_seed(7)
seed = 7


def seed_def(seed=None):
    np_random, seed = seeding.np_random(seed)
    return np_random


META_POLICY_PATH = "/Users/navneetmadhukumar/Downloads/pytorch-maml-rl/saves/maml-halfcheetah-dir/policy-999.pt"
BASELINE_POLICY_PATH = "/Users/navneetmadhukumar/Downloads/pytorch-maml-rl/saves/maml-halfcheetah-dir/baseline.pt"


def sample_tasks(num_tasks, seed=None):
    np_random = seed_def(seed)
    directions = 2 * np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
    tasks = [{'direction': direction} for direction in directions]
    return tasks


def load_meta_learner_params(policy_path, baseline_path, env, num_layers=2):
    policy = torch.load(policy_path)
    baseline = torch.load(baseline_path)

    return policy, baseline


def evaluate(env, policy, max_path_length=100):
    cum_reward = 0
    t = 0
    obs = env.reset()
    for _ in range(max_path_length):
        obs_tensor = torch.from_numpy(obs).to(device='cpu').type(torch.FloatTensor)
        action_tensor = policy(obs_tensor).sample()
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
    policy, baseline = load_meta_learner_params(META_POLICY_PATH, BASELINE_POLICY_PATH, env)
    sampler = BatchSampler(env_name='HalfCheetahDir-v1',
                           batch_size=4000)
    gamma = 0.99
    fast_lr = 0.1
    tau = 1.0
    learner = MetaLearner(sampler, policy, baseline,
                          gamma=gamma,
                          fast_lr=fast_lr, tau=tau,
                          )
    writer = SummaryWriter()

    TEST_TASKS = {'direction': -1.0}
    task = TEST_TASKS
    print(task)
    env.reset_task(task)
    # Sample a batch of transitions
    sampler.reset_task(task)
    num_updates = 5
    num_tasks = 5

    for i in range(ITR):

        episodes = sampler.sample(policy)
        new_params = learner.adapt(episodes)
        policy.load_state_dict(new_params)
        evaluate(env, task, policy)


if __name__ == '__main__':
    main()