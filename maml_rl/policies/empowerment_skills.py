import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import gtimer as gt
import rlkit.rlkit.torch.pytorch_util as ptu
import torch.optim as optim
from rlkit.rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.rlkit.torch.networks import FlattenMlp
from rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from tensorboardX import SummaryWriter
import gym


print("Using the torch version : ", torch.__version__)

EPS = 1E-6


class EmpowermentSkills(TorchRLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 discriminator,
                 q_value_function_1,
                 q_value_function_2,
                 value_function,
                 pool,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 vf_lr=1e-3,
                 disc_lr=1e-3,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 optimizer_class=optim.Adam,

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
                 add_p_z=True,

                 train_policy_with_reparameterization=True,
                 soft_target_tau=1e-2,
                 policy_update_period=1,
                 target_update_period=1,
                 render_eval_paths=False,
                 eval_deterministic=True,

                 eval_policy=None,
                 exploration_policy=None,

                 use_automatic_entropy_tuning=True,
                 target_entropy=None,
                 **kwargs
                 ):

        """

        Uses Soft Actor critic with the twin architecture from TD3 (increased stability in the training)

        :param env:
        :param policy:
        :param discriminator:
        :param q_value_function:
        :param value_function:
        :param pool:
        :param plotter:
        :param lr:
        :param scale_entropy:
        :param discount:
        :param tau:
        :param num_skills:
        :param save_full_state:
        :param find_best_skill_interval:
        :param best_skill_n_rollouts:
        :param learn_p_z:
        :param include_actions:
        :param add_p_z:
        """

        super().__init__(
            env=env,
            exploration_policy=exploration_policy or policy,
            eval_policy=eval_policy,
            **kwargs
        )

        self.env = env
        self.policy = policy
        self.discriminator = discriminator
        self.qvf = q_value_function_1
        self.qvf_2 = q_value_function_2
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
        self.writer = SummaryWriter()

        self.policy_lr = policy_lr
        self.qvf_lr = qf_lr
        self.vf_lr = vf_lr
        self.disc_lr = disc_lr

        self.optim = optimizer_class

        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.render_eval_paths = render_eval_paths

        # Define the target value function
        self.target_vf = self.vf.copy()
        # Define the loss functions for the critic and state value function
        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()

        # Define the optimizers
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qvf.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qvf_2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.disc_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr
        )

        self.action_dim = self.env.action_space.flat_dim
        self.obs_dim = self.env.observation_space.flat_dim

    def sample_empowerment_latents(self):
        """Samples z from p(z), using probabilities in self.p_z."""
        return np.random.choice(self.num_skills, p=self.p_z)

    def split_obs(self, obs):
        obs, z_one_hot = obs[:self.obs_dim], obs[self.obs_dim:]
        return obs, z_one_hot

    def update_critic(self, observation, action, p_z_given, next_observation, done):

        """
        Create minimization operation for the critic Q function.
        :return: TD Loss, Empowerment Reward
        """

        # Get the q value for the observation(obs, z_one_hot) and action.
        q_value_1 = self.qvf(observation, action)
        q_value_2 = self.qvf_2(observation, action)

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
        vf_target_next_obs = self.target_vf(next_observation)
        v_pred = self.vf(observation)

        # Calculate the targets for the Q function (Calculate Q Function Loss)
        q_target = self.reward_scale*empowerment_reward + (1 - done) * self.discount * vf_target_next_obs
        qf1_loss = self.qf_criterion(q_value_1, q_target.detach())
        qf2_loss = self.qf_criterion(q_value_2, q_target.detach())

        return qf1_loss, qf2_loss, empowerment_reward, v_pred

    def update_state_value(self, observation, action, p_z_given,
                           next_observation,
                           done):
        """
        Creates minimization operations for the state value functions.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        :return:
        """

        qf1_loss, qf2_loss, empowerment_reward, _ = self.update_critic(observation=observation,
                                                                       action=action,
                                                                       p_z_given=p_z_given,
                                                                       next_observation=next_observation,
                                                                       done=done)

        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(observation,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        q_new_actions = torch.min(
            self.qvf(observation, new_actions),
            self.qvf_2(observation, new_actions),
        )

        (observation, z_one_hot) = self.split_obs(obs=observation)
        v_pred = self.vf(observation)

        """
               Alpha Loss (if applicable)
               """
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        policy_distribution = self.policy.get_distibution(observation)
        log_pi = policy_distribution.log_pi

        v_target = q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (alpha * log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                        log_pi * (alpha * log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value ** 2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                if self.train_policy_with_reparameterization:
                    policy_loss = (log_pi - q_new_actions).mean()
                else:
                    log_policy_target = q_new_actions - v_pred
                    policy_loss = (
                        log_pi * (log_pi - log_policy_target).detach()
                    ).mean()

                mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
                std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
                pre_tanh_value = policy_outputs[-1]
                pre_activation_reg_loss = self.policy_pre_activation_weight * (
                    (pre_tanh_value**2).sum(dim=1).mean()
                )
                policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
                policy_loss = policy_loss + policy_reg_loss

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Empowerment Reward',
                ptu.get_numpy(empowerment_reward),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        return vf_loss, alpha_loss, alpha, qf1_loss, qf2_loss, empowerment_reward, policy_loss

    def update_discriminator(self, observation, action):
        """

        Creates the minimization operation for the discriminator.

        :return:
        """

        (observation, z_one_hot) = self.split_obs(observation)

        if self.include_actions:
            logits = self.discriminator(observation, action)
        else:
            logits = self.discriminator(observation)

        discriminator_loss = torch.nn.CrossEntropyLoss()(logits, z_one_hot)

        """
        Update the discriminator
        """

        self.disc_optimizer.zero_grad()
        discriminator_loss.backward()
        self.disc_optimizer.step()

        return discriminator_loss

    def _do_training(self, i):

        """

        When training, the policy expects an augmented observation

        obs = observation + num_skills (one hot encoding)

        :return:
        """

        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        p_z = self.sample_empowerment_latents()

        """
        
        Update the networks
        
        """

        rews = np.mean(ptu.get_numpy(rewards))

        vf_loss, alpha_loss, alpha, qf1_loss, qf2_loss, emp_reward, pol_loss = self.update_state_value(
            observation=obs,
            action=actions,
            done=terminals,
            next_observation=next_obs,
            p_z_given=p_z
        )

        # Update the discriminator
        discriminator_loss = self.update_discriminator(observation=obs,
                                                       action=actions)

        disc_loss = np.mean(ptu.get_numpy(discriminator_loss))
        pol = np.mean(ptu.get_numpy(pol_loss))
        emp_rew = np.mean(ptu.get_numpy(emp_reward))
        value_loss = np.mean(ptu.get_numpy(vf_loss))
        q1_loss = np.mean(ptu.get_numpy(qf1_loss))
        q2_loss = np.mean(ptu.get_numpy(qf2_loss))

        self.writer.add_scalar('data/reward', rews, i)
        self.writer.add_scalar('data/policy_loss', pol, i)
        self.writer.add_scalar('data/discriminator_loss', disc_loss, i)
        self.writer.add_scalar('data/empowerment_rewards', emp_rew, i)
        self.writer.add_scalar('data/value_loss', value_loss, i)
        self.writer.add_scalar('data/q_value_loss_1', q1_loss, i)
        self.writer.add_scalar('data/q_value_loss_2', q2_loss, i)

    @property
    def networks(self):
        return [
            self.policy,
            self.qvf,
            self.qvf_2,
            self.vf,
            self.target_vf,
            self.discriminator
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['qf1'] = self.qvf
        snapshot['qf2'] = self.qvf_2
        snapshot['policy'] = self.policy
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        snapshot['discriminator'] = self.discriminator
        return snapshot


if __name__ == '__main__':
    env = gym.make('')
    observation_dim = None
    action_dim = None
    hidden_size = 100
    output_size = 1

    # Define the networks

    q_value_function_1 = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim + action_dim,
        output_size=output_size)

    q_value_function_2 = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim + action_dim,
        output_size=output_size)

    value_function = FlattenMlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=observation_dim,
        output_size=output_size
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[hidden_size, hidden_size],
        obs_dim=observation_dim,
        action_dim=action_dim
    )















