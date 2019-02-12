from architectures import *
from utils import *

env = gym.make("SpaceInvaders-v0")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape
h, w, c = obs_dim
episodes = 100
gamma = 0.99

# common state feature extractor for both policy and value function
state_rep = State_Rep(c, h, w, attn_heads = 2, gru_layers = 2).to(device)

policy = Policy(action_dim, state_rep).to(device)
value_fn = Value_Fn(state_rep).to(device)

policy_optim = optim.Adam(policy.parameters(), lr = 1e-3)
value_fn_optim = optim.Adam(value_fn.parameters(), lr = 1e-3)

train(env, state_rep, policy, value_fn, policy_optim, value_fn_optim, episodes = episodes, gamma = gamma, k1 = 15, k2 = 30)
