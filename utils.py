from includes import *

def train(env, policy, value_fn, policy_optim, value_fn_optim, episodes):
    for episode in range(episodes):
        obs = torch.FloatTensor(env.reset()).to(device)
        rewards = []
        times = []
        state_history = [obs]
        log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)
        while True:
            env.render()

            logits = policy(obs)
            m = torch.distributions.Categorical(logits)
            action = m.sample()
            obs, reward, done, info = env.step(action.item())

            loss = -reward * m.log_prob(action)
            policy_optim.zero_grad()
            loss.backward()
            policy_optim.step()

            obs = torch.FloatTensor(obs)

            if done is True:
                break
