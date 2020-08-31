from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from pprint import pprint

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

#from cnn_model import Actor, Critic
from cnn_model import Actor, Critic
from memory import ReplayBuffer
# from easy_environment import EasyEnvironment
#from tick_environment import TickEnvironment
from ctrl_environment import TickEnvironment

from torchviz import make_dot
torch.autograd.set_detect_anomaly(True)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 2
EPSILON = 1e-6
TAU = 0.01
DEVICE = 'cuda'


class SAC:

    def __init__(self):
        self.action_size = 3
        self.do_evaluation_iterations = True
        self.discount = 0.99
        self.learning_updates_per_learning_session = 1
        self.min_steps_before_learning = 400
        self.tau = 5e-3
        self.update_every_n_steps = 1
        self.device = DEVICE
        self.clip_rewards = True
        self.batch_size = 16
#        self.batch_size = 256
        self.gradient_clipping_norm = 5
        self.automatic_entropy_tuning = True
        self.actor_learning_rate = 3e-4
        self.critic_learning_rate = 3e-4
        self.learning_rate = 5e-3
        self.buffer_size = 1000000

        self.avg_score = 0
        self.avg_score_hist = list()
        self.score_hist = list()
#        self.entropy_weight_term = 1e-3

        self.global_step_number = 0

        self.environment = TickEnvironment(trade_penalty=0.005)

        self.state_space = self.environment.state_space.shape[0]

        self.critic_local_1 = Critic(state_space=self.state_space, device=self.device)
        self.critic_local_2 = Critic(state_space=self.state_space, device=self.device)
        self.critic_optimizer_1 = Adam(self.critic_local_1.parameters(), lr=self.critic_learning_rate, eps=1e-4)
        self.critic_optimizer_2 = Adam(self.critic_local_2.parameters(), lr=self.critic_learning_rate, eps=1e-4)
        self.critic_target_1 = Critic(state_space=self.state_space, device=self.device)
        self.critic_target_2 = Critic(state_space=self.state_space, device=self.device)

        for param in self.critic_target_1.parameters():
            param.requires_grad = False
        for param in self.critic_target_2.parameters():
            param.requires_grad = False

        self.copy_model_over(self.critic_local_1, self.critic_target_1)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=0, device=self.device)

        self.actor_local = Actor(state_space=self.state_space, device=self.device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=self.actor_learning_rate, eps=1e-4)

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate, eps=1e-4)
        else:
            assert False
            self.alpha = self.entropy_weight_term

    def produce_action_and_action_info(self, state):
        """
        Given a state, produce an action, the probability of the action, the log probability of the action
        and the argmax action.

        """

#        state = (state[0].to(self.device), state[1].to(self.device))
        state = state.to(self.device)
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)

        action_distribution = create_actor_distribution('discrete', action_probabilities, self.action_size)
#        action_distribution = Categorical(action_probabilities)

        action = action_distribution.sample().cpu()
#        print(action)
#        print(action)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """
        Calculate the losses for the two critics. This is the ordinary Q-learning loss except the additional
        entropy term is also taken into account.

        """

        with torch.no_grad():
            next_state_action, (action_probabilities,
                                log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target_1(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) -
                                                         2.0 * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.discount * (min_qf_next_target)

#        pprint(action_batch.long())
#        qf1_a = self.critic_local_1(state_batch)
#        qf2_a = self.critic_local_2(state_batch)

#        print(qf1_a)
#        print(action_batch.long())

        qf1 = self.critic_local_1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
#        qf2 = qf2_a.gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """
        Calculate the loss of the actor, including the additional entropy term.

        """

        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local_1(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def save_result(self):
        #        print((self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0)
        if self.episode_number == 1:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extent([np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:])
                                         for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []

    def take_optimization_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False, name=''):
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()
#        network[0].reset()
#        if name == 'a':
#            make_dot(loss).render(view=True)
#            print(len(network))
        loss.backward(retain_graph=retain_graph)

        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()

#        print(f'net: {name}')
#        print(len(network))
#        for net in network:
#            net.reset

    def step(self):
        """
        Run a episode of the environemnt, saving experience.

        """

#        print(f'Step {self.episode_number}')

        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            #            print(f'\t{self.global_step_number} {len(self.environment)}')
            #            print(self.episode_step_number_val, len(self.environment))
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.learning_updates_per_learning_session):
                    self.learn()
#                    print(f'time for learn {self.episode_step_number_val}')
            mask = False if self.episode_step_number_val + 2 >= len(self.environment) else self.done
            if not eval_ep:
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
#        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        if state is None:
            state = self.state

        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)

        elif self.global_step_number < self.min_steps_before_learning:
            action = self.environment.action_space.sample()
#            print('Picking random action ', action)
        else:
            action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        if state is None:
            state = self.state

#        state = (state[0].to(self.device), state[1].to(self.device))
        state = state.to(self.device)
#        state = torch.FloatTensor([state[0]]).to(self.device),
#        torch.FloatTensor

        if len(state[0].shape) == 1:
            state = state.unsqueeze(0)
        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def calculate_entropy_tuning_loss(self, log_pi):
        """
        Calculate the loss of the entropy temperature parameter.

        """
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def learn(self):
        #        print('learning')
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
#        print(type(state_batch))
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch,
                                                          action_batch,
                                                          reward_batch,
                                                          next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)

        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
#            self.take_optimization_step(self.alpha_optim, None, alpha_loss, None)
#            self.alpha = self.log_alpha.exp()
        else:
            assert False
            alpha_loss = None

        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """
        Updates the parameters of the actor, the two critics, and the entropy parameter.

        """

#        self.critic_local_1.reset()
#        self.critic_local_2.reset()
#        self.actor_local.reset()

        self.take_optimization_step(self.critic_optimizer_1,
                                    self.critic_local_1,
                                    critic_loss_1,
                                    self.gradient_clipping_norm, name='c1')
        self.take_optimization_step(self.critic_optimizer_2,
                                    self.critic_local_2,
                                    critic_loss_2,
                                    self.gradient_clipping_norm, name='c2')
        self.take_optimization_step(self.actor_optimizer,
                                    self.actor_local,
                                    actor_loss,
                                    self.gradient_clipping_norm, name='a')

        self.soft_update_of_target_network(self.critic_local_1,
                                           self.critic_target_1,
                                           self.tau)
        self.soft_update_of_target_network(self.critic_local_2,
                                           self.critic_target_2,
                                           self.tau)

        if alpha_loss is not None:
            self.take_optimization_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def save_experience(self, memory=None, experience=None):
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def sample_experiences(self):
        return self.memory.sample()

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.batch_size

    def conduct_action(self, action):
        """
        Conduct an action in the environment.

        """

        self.next_state, self.reward, self.done = self.environment.step(action)
#        print(self.environment.idx, self.done)
        self.total_episode_score_so_far += self.reward
        if self.clip_rewards:
            self.reward = np.clip(self.reward, -1.0, 1.0)

    def time_for_critic_and_actor_to_learn(self):
        """
        Returns a true if there are enough experiences to learn from, and it is time for the actor and
        critic to learn.

        """
        return self.global_step_number > self.min_steps_before_learning and \
            self.enough_experiences_to_learn_from() and \
            self.global_step_number % self.update_every_n_steps == 0

    def run_n_episodes(self, num_episodes=None):

        self.episode_number = 0

        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()

    def print_summary_of_latest_evaluation_episode(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        env = self.environment
        ax1.plot(np.arange(len(env.prices)), env.prices, linewidth=0.3, color='black')

        for t0, t1 in env.hold_intervals:
            ax1.plot(np.arange(t0, t1 + 1), env.prices[t0:t1 + 1], color='green', linewidth=0.4)
        for t0, t1 in env.flat_intervals:
            ax1.plot(np.arange(t0, t1 + 1), env.prices[t0:t1 + 1], color='red', linewidth=0.4)

        if len(env.buy_pts) > 0:
            bpts_t, bpts_x = tuple(zip(*env.buy_pts))
            bpts_x = tuple(x.item() for x in bpts_x)
            ax1.scatter(bpts_t, bpts_x, s=4, color='green')
        if len(env.sell_pts) > 0:
            spts_t, spts_x = tuple(zip(*env.sell_pts))
            spts_x = tuple(x.item() for x in spts_x)
            ax1.scatter(spts_t, spts_x, s=4, color='red')

        if self.episode_number == 0:
            self.avg_score = self.total_episode_score_so_far
        else:
            self.avg_score = (1 - 0.05) * self.avg_score + 0.05 * self.total_episode_score_so_far
        self.avg_score_hist.append(self.avg_score)
        self.score_hist.append(self.total_episode_score_so_far)

        ax2.plot(np.arange(len(self.score_hist)), self.score_hist, linewidth=0.2)
        ax2.plot(np.arange(len(self.score_hist)), self.avg_score_hist, linewidth=0.5)

        ax3.plot(np.arange(min(200, len(self.score_hist))), self.score_hist[-200:], linewidth=0.2)
        ax3.plot(np.arange(min(200, len(self.score_hist))), self.avg_score_hist[-200:], linewidth=0.5)

        plt.show()
        print('-' * 20)
        print('Episode {}, score {} '.format(self.episode_number, self.total_episode_score_so_far))
        print('Buy: {}, sell: {}, net: {}'.format(len(env.buy_pts), len(env.sell_pts), env.net))
        print('s: {}, n: {}, l: {}'.format(env.raw_short_count, env.raw_neutral_count, env.raw_long_count))
        print(f'alpha: {self.alpha.item()}')
        print('-' * 20)
        print(f'Run: {env.idx}')

    def soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def copy_model_over(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())


def create_actor_distribution(action_types, actor_output, action_size):
    if action_types == 'discrete':
        assert actor_output.size()[1] == action_size, 'actor output wrong size'
        action_distribution = Categorical(actor_output)
    return action_distribution


agent = SAC()

agent.run_n_episodes(10000000)
