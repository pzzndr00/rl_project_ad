# Contains the agents definition (and what is needed for training) for both vanilla and duelling DQN

import torch
import torch.nn as nn
import random
import numpy as np

# my modules 
import memoryBuffer as mb

# Vanilla DQN agent class
class DQN_agent(nn.Module):
    """Vanilla DQN agent"""
    
    def __init__(self, input_size:int, output_size:int, discount_factor = 0.8, loss_function = nn.MSELoss(), lr = 5e-4,  hidden_size1:int = 128, hidden_size2:int = 128, device = 'cpu'):
        """
        Vanilla DQN __init__ function (2 layers Q network)

        Parameters
        __________

        input_size: (int) input size of the DQN network
        output_size:  (int) output size of the DQN network
        
        discount_factor: (optional) discount factor
        loss_function: (optional) loss function for Q-network optimization
        lr: (optional) learning rate
        hidden_size1: (int) (optional) first hidden layer size
        hidden_size2: (int) (optional) second hidden layer size
        device: (optional) pytorch device
        """
        
        super(DQN_agent, self).__init__()

        # layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.act = nn.LeakyReLU()

        # parameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = lr)
        self.loss_function = loss_function
        self.discount_factor = discount_factor
        self.device = device
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        return out

    def training_step(self, Q_hat, replay_buffer:mb.memory, batch_size:int = 128):
        """
        Training step function

        Parameters
        __________

        Q_hat: Q network used for estimation of action values
        replay_buffer: (mb.memory) replay buffer containing the gathered experience

        batch_size: (int) (optional) size of the batch that will used for one training step

        """
        device = self.device

        # batch sampling and conversion to tensors #############################
        batch = replay_buffer.sample(batch_size = batch_size)

        batch_tr = mb.Transition(*zip(*batch))

        state_batch_tensor = torch.cat(batch_tr.state).reshape((-1, self.input_size)).to(device=device).to(torch.float) # for each row a step       
        action_batch = np.array(batch_tr.action) # row vector containing the selected action for every sampled transition
        reward_batch_tensor = torch.tensor(batch_tr.reward).to(device=device).to(torch.float) # row vector
        next_state_batch_tensor = torch.cat(batch_tr.next_state).reshape((-1, self.input_size)).to(device=device).to(torch.float) # for each row a step  

        done_batch_tensor = torch.tensor(batch_tr.done).to(device=device) # row vector of zeros and ones 
        done_float_batch_tensor = done_batch_tensor.to(torch.float)

        # selected actions Q values
        Q_vals = self(state_batch_tensor)
        Q_vals = Q_vals[torch.arange(Q_vals.size(0)), action_batch] # row vector with values only for selected actions


        # max Q values for next state
        with torch.no_grad():
            max_Q_hat_vals_next = torch.max(self(next_state_batch_tensor), dim=1)[0] # row vector

        y = reward_batch_tensor + self.discount_factor * max_Q_hat_vals_next * (1-done_float_batch_tensor)

        # computation of the loss and training step
        loss = self.loss_function(Q_vals, y)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss

    def act_eps_greedy(self, state_tensor:torch.tensor, eps):
        """
        Epsilon greedy action selection used for training

        Parameters
        __________

        state_tensor: (torch.tensor) row tensor containg the vectorized state tensor
        eps: epsilon

        Returns
        _______
        greedy action with probability 1 - eps, random action with probability eps

        """
        actions_values = self(state_tensor)

        u = random.random()

        if u < 1 - eps:
            return torch.argmax(actions_values).cpu()
        else:
            return random.choice(range(self.output_size))

    def act_greedy(self, state_tensor):
        """
        Greedy action selection

        Parameters
        __________

        state_tensor: (torch.tensor) row tensor containg the vectorized state tensor

        Returns
        _______
        greedy action
        
        """
        actions_values = self(state_tensor)
        return torch.argmax(actions_values)
