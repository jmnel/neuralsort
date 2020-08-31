from collections import namedtuple, deque
import random
import torch
import numpy as np


class ReplayBuffer():

    def __init__(self, buffer_size, batch_size, seed, device='cpu'):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
#        self.device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
        self.device = device

    def add_experience(self, states, actions, rewards, next_states, dones):
        """
        Add experience(s) into the replay buffer.
        """

        if type(dones) == list:
            assert type(dones[0]) != list, 'A done should not be a list'
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actinos, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, seperate_out_data_types=True):
        """
        Draw a random sample of experience from the replay buffer.
        """

        experiences = self.pick_experiences(num_experiences)
        if seperate_out_data_types:
            states, actions, rewards, next_states, dones = self.seperate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def seperate_out_data_types(self, experiences):
        """
        Puts the sampled experience into the correct format for a PyTorch neural network.
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
