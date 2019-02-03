from architectures import *
from utils import *

env = gym.make("SpaceInvaders-v0")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape
h, w, c = obs_dim
episodes = 100
gamma = 0.99

policy = Policy(action_dim, c, h, w).to(device)
value_fn = Value_Fn(c, h, w).to(device)

policy_optim = optim.Adam(policy.parameters(), lr = 1e-3)
value_fn_optim = optim.Adam(value_fn.parameters(), lr = 1e-3)

train(env, policy, value_fn, policy_optim, value_fn_optim, episodes = episodes, gamma = gamma)
