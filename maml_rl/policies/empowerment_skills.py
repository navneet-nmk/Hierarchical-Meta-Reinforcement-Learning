import numpy as np
import torch


print("Using the torch version : ", torch.__version__)

EPS = 1E-6


class EmpowermentSkills(object):

    def __init__(self,
                 env,
                 policy,
                 discriminator,
                 q_value_function,
                 value_function,
                 pool,
                 plotter=None,
                 lr=3E-3,
                 scale_entropy=1,
                 discount=0.99,
                 tau=0.01,
                 num_skills=20,
                 save_full_state=False,
                 find_best_skill_interval=10,
                 best_skill_n_rollouts=10,
                 learn_p_z=False,
                 include_actions=False,
                 add_p_z=True
                 ):

        self.env = env
        self.policy = policy
        self.discriminator = discriminator
        self.qvf = q_value_function
        self.vf = value_function
        self.pool = pool
        self.plotter = plotter
        self.lr = lr
        self.scale_entropy = scale_entropy
        self.discount = discount
        self.tau = tau
        self.num_skills = num_skills
        self.save_full_state = save_full_state
        self.find_best_skill_interval = find_best_skill_interval
        self.best_skill_n_rollouts = best_skill_n_rollouts
        self.learn_p_z = learn_p_z
        self.include_actions = include_actions
        self.add_p_z = add_p_z
        self.p_z = np.full(num_skills, 1.0 / num_skills)

        """
                Args:
                    base_kwargs (dict): dictionary of base arguments that are directly
                        passed to the base `RLAlgorithm` constructor.
                    env (`rllab.Env`): rllab environment object.
                    policy: (`rllab.NNPolicy`): A policy function approximator.
                    discriminator: (`rllab.NNPolicy`): A discriminator for z.
                    qf (`ValueFunction`): Q-function approximator.
                    vf (`ValueFunction`): Soft value function approximator.
                    pool (`PoolBase`): Replay buffer to add gathered samples to.
                    plotter (`QFPolicyPlotter`): Plotter instance to be used for
                        visualizing Q-function during training.
                    lr (`float`): Learning rate used for the function approximators.
                    scale_entropy (`float`): Scaling factor for entropy.
                    discount (`float`): Discount factor for Q-function updates.
                    tau (`float`): Soft value function target update weight.
                    num_skills (`int`): Number of skills/options to learn.
                    save_full_state (`bool`): If True, save the full class in the
                        snapshot. See `self.get_snapshot` for more information.
                    find_best_skill_interval (`int`): How often to recompute the best
                        skill.
                    best_skill_n_rollouts (`int`): When finding the best skill, how
                        many rollouts to do per skill.
                    include_actions (`bool`): Whether to pass actions to the
                        discriminator.
                    add_p_z (`bool`): Whether th include log p(z) in the pseudo-reward.
        """

        self.action_dim = self.env.action_space.flat_dim
        self.obs_dim = self.env.observation_space.flat_dim

    def sample_empowerment_latents(self):
        """Samples z from p(z), using probabilities in self.p_z."""
        return np.random.choice(self.num_skills, p=self.p_z)

    def split_obs(self, obs):
        return obs, obs

    def update_critic(self, observation, action, p_z_given, done):

        """
        Create minimization operation for the critic Q function.
        :return:
        """

        q_value = self.qvf(observation, action)
        (observation, z_one_hot) = self.split_obs(obs=observation)

        if self.include_actions:
            logits = self.discriminator(observation, action)
        else:
            logits = self.discriminator(observation)

        # The empowerment reward is defined as the cross entropy loss between the
        # true skill and the selected skill.
        empowerment_reward = -1 * torch.nn.CrossEntropyLoss()(z_one_hot, logits)

        p_z = torch.sum(p_z_given*z_one_hot, axis=1)
        log_p_z = torch.log(p_z+EPS)

        if self.add_p_z:
            empowerment_reward -= log_p_z

        # Now we will calculate the value function and critic Q function update.
        vf_next_target = self.vf(observation)

        # Calculate the targets for the Q function
        with torch.no_grad():
            ys = empowerment_reward + (1 - done) * self.discount * vf_next_target

        ys.detach()
        temporal_difference_loss = 0.5*torch.nn.MSELoss()(ys, q_value)

        return temporal_difference_loss, empowerment_reward

    









