import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Net_actor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size1 = 128, hidden_size2 =128, device = 'cpu'):
        """
        Two hidden layers FFNN, soft max final layer

        """
        super(Net_actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size1, device = self.device)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, device = self.device)
        self.fc3 = nn.Linear(hidden_size2, output_size, device = self.device)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=-1)  # Softmax over the last dimension
        return out

class Net_critic(nn.Module):

    def __init__(self, input_size, output_size = 1, hidden_size1 = 128, hidden_size2 =128, device = 'cpu'):
        """
        Two hidden layers FFNN

        """
        super(Net_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size1, device = self.device)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, device = self.device)
        self.fc3 = nn.Linear(hidden_size2, output_size, device = self.device)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Agent:
    def __init__(self, env: highway_env, batch_size=256, discount = 0.99, clip_eps = 0.1, step_size = 1e-4, actor_rep = 15, critic_rep = 5, device = 'cpu'):
        self.discount = discount
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep
        
        features = features_lables = highway_env.envs.common.observation.KinematicObservation.FEATURES
        self.actor = Net_actor(input_size = 25, output_size = env.env.action_space.n, device=device)
        
        self.critic = Net_critic(input_size = 25, device = device)

        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr = step_size)



    def advantageActorCritic(self, env):

        return


def policy(actions, env,  eps = 0.9):
    """
    epsilon-greedy
    """
    u = random.uniform(0, 1)
    if u < eps: return actions.argmax()
    else: return random.choice(env.unwrapped.get_available_actions())
