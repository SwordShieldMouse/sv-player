from algs import *

env = gym.make("SpaceInvaders-v0")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape

episodes = 100
T = 1000
