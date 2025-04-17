import torch
import torch.nn as nn
import random
import numpy as np

# my modules
import memoryBuffer as mb

class PPO_actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1 = 128, hidden_size2 = 128):
        super(PPO_actor, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1) # first hidden layer
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2) # second hidden layer
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size) # output layer

        self.activation = nn.ReLU() # hidden layers activation function
        self.out_act = nn.Softmax(dim=-1)

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        out = self.out_act(self.fc3(x)) # softmax output activation
        return out

class PPO_critic(nn.Module):
    def __init__(self, input_size, hidden_size1 = 128, hidden_size2 = 128):
        super(PPO_critic, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1) # first hidden layer
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2) # second hidden layer
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features = 1) # output layer

        self.activation = nn.ReLU() # hidden layers activation function

        self.input_size = input_size
        self.output_size = 1
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        out = self.fc3(x) # identity output activation
        return out


class clip_loss(nn.Module):

    def __init__(self):
        super(clip_loss, self).__init__()

    def forward(self, td_error, importance_sampling_ratio, clip_eps):

        loss_actor = torch.minimum(
                td_error * importance_sampling_ratio, 
                td_error * torch.clip(importance_sampling_ratio, min = 1-clip_eps, max = 1+clip_eps)
                )

        loss_actor = -torch.mean(loss_actor)

        return loss_actor


class PPO_agent(nn.Module):

    def __init__(self, state_size, actions_set_cardinality, env, discount_factor = 0.99, clip_eps = 0.2, lr = 1e-4, actor_rep = 15, critic_rep= 5, device = torch.device('cpu')):
        super(PPO_agent, self).__init__()

        self.device = device
        self.discount_factor = discount_factor
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep

        self.actor = PPO_actor(input_size=state_size, output_size = actions_set_cardinality) # actor model
        self.actor.to(self.device)
        self.actor_loss_fcn = clip_loss()

        self.critic = PPO_critic(input_size=state_size)  # critic model
        self.critic.to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr = 5*lr)

        self.input_size = state_size

        self.output_size = (self.actor.output_size, self.critic.output_size)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def training_step(self, batch_size:int,  replay_buffer:mb.memory, critic_loss_fcn = nn.MSELoss()):
        
        device = self.device

        batch_size = min(batch_size, len(replay_buffer))

        # batch sampling and conversion to tensors #############################
        batch = replay_buffer.sample(batch_size = batch_size)

        batch_tr = mb.Transition(*zip(*batch))

        state_batch_tensor = torch.cat(batch_tr.state).reshape((-1, self.input_size)).to(device=device).to(torch.float) # for each row a step       
        action_batch = np.array(batch_tr.action) # row vector containing the selected action for every sampled transition
        reward_batch_tensor = torch.tensor(batch_tr.reward).to(device=device).to(torch.float) # row vector
        next_state_batch_tensor = torch.cat(batch_tr.next_state).reshape((-1, self.input_size)).to(device=device).to(torch.float) # for each row a step  

        done_batch_tensor = torch.tensor(batch_tr.done).to(device=device) # row vector of zeros and ones 
        done_float_batch_tensor = done_batch_tensor.to(torch.float)
        #########################################################################

        states_vals_cr = self.critic(state_batch_tensor).reshape(-1) # column vector to row vector
        next_states_vals_cr = self.critic(next_state_batch_tensor).reshape(-1) # column vector to row vector

        with torch.no_grad():
            initial_action_probs_ac = self.actor(state_batch_tensor)[np.arange(batch_size), action_batch] # row vector with selected action pi_old(at|st)

        # computes rewards to go using TD(0) target

        rewards_to_go = reward_batch_tensor + self.discount_factor*next_states_vals_cr*(1-done_float_batch_tensor)

        td_error = rewards_to_go - states_vals_cr # used as an approximation of the advantage
        
        # actor parameters oprtimization

        for _ in range(self.actor_rep):

            action_probs_ac = self.actor(state_batch_tensor)[np.arange(batch_size), action_batch]

            importance_sampling_ratio = action_probs_ac / initial_action_probs_ac

            loss_actor = self.actor_loss_fcn(td_error=td_error, importance_sampling_ratio=importance_sampling_ratio, clip_eps = self.clip_eps)
            
            loss_actor.backward(retain_graph=True)
            self.actor_opt.zero_grad()
            
            self.actor_opt.step()



        # critic parameters optimization
        for _ in range(self.critic_rep):

            states_vals_cr = self.critic(state_batch_tensor).reshape(-1) # column vector to row vector
            next_states_vals_cr = self.critic(next_state_batch_tensor).reshape(-1) # column vector to row vector
            
            rewards_to_go = reward_batch_tensor + self.discount_factor*next_states_vals_cr*(1-done_float_batch_tensor)

            loss_critic = critic_loss_fcn(states_vals_cr, rewards_to_go)
            loss_critic.backward()
            self.critic.zero_grad()
            
            self.critic_opt.step()

        return loss_actor, loss_critic

    def act(self, state_tensor:torch.tensor):
        """

        Function that choses an action according to the output probabilities given by the actor for the given state
        PARAMETERS
        ----------
        state_tensor: a pytorch tensor containing the vectorized state of highway-env

        RETURNS
        -------
        action: the selected action
        pi_given_s: the output of the ctitic (pyTorch tensor) given the current state 

        """

        with torch.no_grad():
            pi_given_s = self.actor(state_tensor)

        action = random.choices( np.arange(stop = pi_given_s.shape[0]) , weights=pi_given_s.cpu().detach().numpy(), k = 1)[0]


        return action, pi_given_s

    

        
            

