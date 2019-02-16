from includes import *

class Buffer():
    def __init__(self, length):
        self.length = length
        self.buffer = [Variable(torch.zeros(1), requires_grad = True).to(device)] * length
        self.replace_ix = 0 # to keep track of which position in array we should be replacing with a new element

    def push(self, x):
        self.buffer[self.replace_ix] = x
        if self.replace_ix + 1 == self.length:
            self.replace_ix = 0
        else:
            self.replace_ix += 1

    def sum(self):
        return torch.sum(torch.stack(self.buffer).to(device), dim = 0)


def train(env, state_rep, policy, value_fn, policy_optim, value_fn_optim, episodes, gamma, k1, k2):
    for episode in range(episodes):
        obs = torch.FloatTensor(env.reset()).to(device)
        hidden = state_rep.reset_hidden()

        obs_rep, hidden = state_rep(obs, hidden)

        reward_history = []
        times = []
        state_history = [obs_rep]
        log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)

        # RNN implemented with truncated backpropagation through time
        # every k1 time-steps, backpropagate back k2 steps

        t = 0

        # maintain a buffer of losses of length k2
        policy_loss_buffer = Buffer(k2)
        value_loss_buffer = Buffer(k2)

        while True:
            env.render()

            logits = policy(obs_rep)
            m = torch.distributions.Categorical(logits)
            action = m.sample()
            obs, reward, done, info = env.step(action.item())
            obs = torch.FloatTensor(obs).to(device)

            # maintain memory of states
            # TODO: make sure to delete memory when we go on for too long
            obs_rep, hidden = state_rep(obs, hidden)
            state_history.append(obs_rep)
            reward_history.append(reward)


            # backpropagate if we have gone for k1 steps since last backprop
            if t % k1 == 0 and t > 0:
                # get past k2 losses
                policy_loss = policy_loss_buffer.sum()
                value_loss = value_loss_buffer.sum()

                policy_optim.zero_grad()
                policy_loss.backward(retain_graph = True) # need to retain graph because policy and value_fn share state_rep
                policy_optim.step()

                value_fn_optim.zero_grad()
                value_loss.backward()
                value_fn_optim.step()

                hidden = state_rep.reset_hidden()

                #print("TBPTT successful")


            # optimize the policy
            # use the advantage function as the reward
            policy_loss = -(reward + gamma * value_fn(state_history[-1]) - value_fn(state_history[-2])) * m.log_prob(action)
            #print(policy_loss.squeeze(0).shape)
            policy_loss_buffer.push(policy_loss.squeeze(0))
            #policy_optim.zero_grad()
            #policy_loss.backward(retain_graph = True) # need to retain graph because policy and value_fn share state_rep
            #policy_optim.step()

            # optimize the value function with a semi-gradient method
            next_value = value_fn(state_history[-1]).detach() # detach because we want to do semi-gradient
            value_loss = (reward + gamma * next_value - value_fn(state_history[-2])) ** 2
            value_loss_buffer.push(value_loss.squeeze(0))
            #value_fn_optim.zero_grad()
            #value_loss.backward()
            #value_fn_optim.step()

            t += 1

            if done is True:
                print("episode {} is done".format(episode))
                print("reward is {}".format(np.sum(reward_history)))
                break
