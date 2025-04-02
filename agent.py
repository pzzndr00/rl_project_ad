import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class agent_model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size1 = 128, hidden_size2 =128):
        """
        Two hidden layers FFNN, soft max final layer
        

        """
        super(agent_model, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=-1)  # Softmax over the last dimension
        return out


def eps_greedy_policy(actions_values, available_actions, eps):

    u = random.random()

    if u < eps:
        return np.argmax(actions_values)
    else:
        return random.choice(available_actions)

        