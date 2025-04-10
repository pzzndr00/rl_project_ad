import torch
import torch.nn as nn
# import torch.nn.functional as F
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

        state_batch = torch.cat(batch_tr.state).reshape((-1, self.input_size))       
        action_batch = np.array(batch_tr.action)
        reward_batch = np.array(batch_tr.reward)
        next_state_batch = torch.cat(batch_tr.next_state).reshape((-1, self.input_size))
        done_batch = list(batch_tr.done)
        
        # moving tensors to device
        state_batch = state_batch.to(device=device)
        next_state_batch = next_state_batch.to(device=device)

        Q_values = torch.zeros(batch_size)
        Q_values = Q_values.to(device = device)

        y = torch.zeros(batch_size)
        y = y.to(device=device)
 
        for i in range(batch_size):

            if done_batch[i]:
                y[i] = reward_batch[i]
            else:
                Q_hat_next = Q_hat(next_state_batch[i])
                max_Q_hat_next = Q_hat_next.max().item()

                y[i] = reward_batch[i] + self.discount_factor * max_Q_hat_next

            # for each sample compute the current agent action values
            Q_values[i] = self(state_batch[i])[action_batch[i]]
        
        # computation of the loss and training step
        loss = self.loss_function(Q_values, y)

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