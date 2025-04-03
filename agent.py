import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Q_network(nn.Module):

    def __init__(self, input_size, output_size, hidden_size1 = 128, hidden_size2 =128):
        """
        Two hidden layers FFNN, soft max final layer
        

        """
        super(Q_network, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = self.fc3(x)
        return out


    def act_greedely(self, state):
        raise NotImplemented
        return
    
def eps_greedy_policy(actions_values, available_actions, eps):

    u = random.random()

    if u < 1 - eps:
        return np.argmax(actions_values)
    else:
        return random.choice(available_actions)

        