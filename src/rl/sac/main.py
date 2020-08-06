import torch

from environment5 import Environment
from replay_memory import ReplayMemory
from sac import SAC

env = Environment()

agent = SAC(env.observation_space.shape[0], env.action_space)

memory = ReplayMemory(10)

total_numpsteps = 0
updates = 0

for i_episode in range(10):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
