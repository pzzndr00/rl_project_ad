import torch
import torch.nn as nn
import random
import numpy as np

# my modules 
import memoryBuffer as mb

class DQN_agent(nn.Module):

    def __init__(self, input_size, output_size, discount_factor = 0.8, loss_function = nn.MSELoss(), lr = 5e-4,  hidden_size1 = 128, hidden_size2 = 128, device = 'cpu'):
        super(DQN_agent, self).__init__()

        # layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.act = nn.ReLU()

        # parameters
        # self.model = DQN_network(input_size, output_size, hidden_size1 = hidden_size1, hidden_size2 = hidden_size2)
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


    def training_step(self, Q_hat, replay_buffer:mb.memory, batch_size = 128):
        
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

    def act_eps_greedy(self, state_tensor, eps):
        
        actions_values = self(state_tensor)

        u = random.random()

        if u < 1 - eps:
            return torch.argmax(actions_values).cpu()
        else:
            return random.choice(range(self.output_size))

        return

    def act_greedy(self, state_tensor):
        actions_values = self(state_tensor)
        return torch.argmax(actions_values)