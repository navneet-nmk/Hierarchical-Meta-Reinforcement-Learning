from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.rlkit.torch.pytorch_util as ptu
from rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.rlkit.torch.torch_rl_algorithm import TorchTrainer
from tensorboardX import SummaryWriter

print("Using the torch version : ", torch.__version__)

EPS = 1E-6


class EmpowermentSkillsTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            higher_level_policy,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            value_function,
            discriminator,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            value_lr=1e-3,
            discriminator_lr=1e-3,
            optimizer_class=optim.Adam,

            lr=3E-3,
            scale_entropy=1,
            tau=0.01,
            num_skills=50,
            save_full_state=False,
            find_best_skill_interval=10,
            best_skill_n_rollouts=10,
            learn_p_z=False,
            include_actions=False,
            add_p_z=True,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            train_policy_with_reparameterization=True,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            policy_update_period=1,
    ):
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.discriminator = discriminator
        self.value_network = value_function
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.higher_policy = higher_level_policy
        # Define the target value function
        self.target_vf = self.value_network.copy()
        self.train_policy_with_reparameterization = train_policy_with_reparameterization

        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.writer = SummaryWriter()
        self.tau = tau
        self.num_skills = num_skills
        self.save_full_state = save_full_state
        self.find_best_skill_interval = find_best_skill_interval
        self.best_skill_n_rollouts = best_skill_n_rollouts
        self.learn_p_z = learn_p_z
        self.include_actions = include_actions
        self.add_p_z = add_p_z
        self.p_z = np.full(num_skills, 1.0 / num_skills)

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

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discriminator_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=discriminator_lr
        )

        self.vf_optimizer = optimizer_class(
            self.value_network.parameters(),
            lr=value_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.action_dim = self.env.action_space
        self.obs_dim = self.env.observation_space

    def sample_empowerment_latents(self, observation):
        """Samples z from p(z), using probabilities in self.p_z."""
        return self.higher_policy(observation).sample()

    def split_obs(self, obs):
        obs, z_one_hot = obs[:self.obs_dim], obs[self.obs_dim:]
        return obs, z_one_hot

    def update_critic(self, observation, action, p_z_given, next_observation, done):

        """
        Create minimization operation for the critic Q function.
        :return: TD Loss, Empowerment Reward
        """

        # Get the q value for the observation(obs, z_one_hot) and action.
        q_value_1 = self.qf1(observation, action)
        q_value_2 = self.qf2(observation, action)

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
        v_pred = self.value_network(observation)

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
            self.qf1(observation, new_actions),
            self.qf2(observation, new_actions),
        )

        (observation, z_one_hot) = self.split_obs(obs=observation)
        v_pred = self.value_network(observation)

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
                self.value_network, self.target_vf, self.soft_target_tau
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

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        p_z = self.sample_empowerment_latents(obs)

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

        i = self._n_train_steps_total

        self.writer.add_scalar('data/reward', rews, i)
        self.writer.add_scalar('data/policy_loss', pol, i)
        self.writer.add_scalar('data/discriminator_loss', disc_loss, i)
        self.writer.add_scalar('data/empowerment_rewards', emp_rew, i)
        self.writer.add_scalar('data/value_loss', value_loss, i)
        self.writer.add_scalar('data/q_value_loss_1', q1_loss, i)
        self.writer.add_scalar('data/q_value_loss_2', q2_loss, i)

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.target_vf,
            self.discriminator,
            self.value_network
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            value=self.value_network,
            discriminator=self.discriminator,
            target_value=self.target_vf,
        )