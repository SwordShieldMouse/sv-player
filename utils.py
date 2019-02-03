from includes import *

def train(env, policy, value_fn, policy_optim, value_fn_optim, episodes, gamma):
    # implement actor-critic
    for episode in range(episodes):
        obs = torch.FloatTensor(env.reset()).to(device)
        reward_history = []
        times = []
        state_history = [obs]
        log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)
        while True:
            env.render()

            logits = policy(obs)
            m = torch.distributions.Categorical(logits)
            action = m.sample()
            obs, reward, done, info = env.step(action.item())
            obs = torch.FloatTensor(obs)

            # maintain memory of states
            # TODO: make sure to delete memory when we go on for too long
            state_history.append(obs)
            reward_history.append(reward)

            # optimize the policy
            # use the advantage function as the reward
            policy_loss = -(reward + gamma * value_fn(state_history[-1]) - value_fn(state_history[-2])) * m.log_prob(action)
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # optimize the value function with a semi-gradient method
            next_value = value_fn(state_history[-1]).detach() # detach because we want to do semi-gradient
            value_loss = (reward + gamma * next_value - value_fn(state_history[-2])) ** 2
            value_fn_optim.zero_grad()
            value_loss.backward()
            value_fn_optim.zero_grad()

            if done is True:
                print("episode {} is done".format(episode))
                print("reward is {}".format(np.sum(reward_history)))
                break
