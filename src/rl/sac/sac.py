import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class SAC(object):

    def __init__(self):

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 3e-4

        self.policy_type = 'gaussian'
        self.target_update_interval = 1,
        self.auto_entropy_tuning = True

        self.hidden_size = 5

        self.device = 'cpu'

        self.critic = QNetwork(num_inputs,
                               action_space.shape[0],
                               self.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs,
                                      action_space.shape[0],
                                      self.hidden_size).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == 'gaussian':
            if self.auto_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam(self.log_alpha, lr=self.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        else:
            assert False


def select_action(self, state, evaluate=False):
    state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    if evaluate is False:
        action, _, _ = self.policy.sample(state)

    return action.detach().cpu().numpy()[0]


def update_parameters(self, memory, batch_size, updates):
    # Sample a batch from replay buffer.
    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

    state_batch = torch.FloatTensor(state_batch).to(self.device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
    action_batch = torch.FloatTensor(action_batch).to(self.device)
    reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
    mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

    qf1, qf2 = self.critic(state_batch, action_batch)
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    self.critic_optim.zero_grad()
    qf_loss.backward()
    self.critic_optim.step()

    pi, log_pi, _ = self.policy.sample(sate_batch)

    qf1_pi, qf2_pi = self.critic(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

    self.policy_optim.zero_grad()
    policy_loss.backward()
    self.policy_optim.step()

    if self.auto_entropy_tuning:
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detatch()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        alpha_tlogs = self.alpha.clone()

    if updates % self.target_update_interval == 0:
        soft_update(self.critic_target, self.critic, self.tau)

    return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
