import torch
import torch.nn as nn
import random
import numpy as np


class trj_set(object):
    def __init__(self):
        super(trajectory, self).__init__()

        self.states = []        # S
        self.actions = []       # A
        self.rewards = []       # R
        self.next_states = []   # S' redoundant but helps in training

        self.probs = [] # pi(a|s)
        
        self.done = []

    def append_transition(self, state, action, reward, next_state, probs, done):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

        self.probs.append(probs)

        self.done.append(done)

    def __len__(self):
        return len(self.states)




class PPO_actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1 = 128, hidden_size2 = 128):
        super(PPO_actor, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1) # first hidden layer
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2) # second hidden layer
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size) # output layer

        self.act = nn.ReLU() # hidden layers activation function

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        out = nn.Softmax(self.fc3(x)) # softmax output activation
        return out

class PPO_critic(nn.Module):
    def __init__(self, input_size, hidden_size1 = 128, hidden_size2 = 128):
        super(PPO_critic, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1) # first hidden layer
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2) # second hidden layer
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features = 1) # output layer

        self.act = nn.ReLU() # hidden layers activation function

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        out = self.fc3(x) # identity output activation
        return out


class PPO_agent(object):

    def __init__(self, state_size, actions_set_cardinality, env, discount_factor = 0.8, clip_eps = 0.1, lr = 1e-4, actor_rep = 15, critic_rep= 5, device = torch.device('cpu')):
        super(PPO_agent, self).__init__()

        self.device = device
        self.discount_factor = discount_factor
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep

        self.actor = PPO_actor(input_size=state_size, output_size = actions_set_cardinality) # actor model
        self.actor.to(self.device)

        self.critic = PPO_critic(input_size=state_size)  # critic model
        self.critic.to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr = 3*lr)



    def training_step(self, trj_list:trj_list, loss_function = nn.MSELoss()):
        
        raise NotImplementedError

    

        
            

