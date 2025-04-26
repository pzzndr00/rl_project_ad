import torch
import torch.nn as nn
import random
import numpy as np

# my modules

class PPO_Buffer():
    def __init__(self):

        self.state_tesnsors_list = []
        self.action_list = []
        self.action_probs_list = []
        self.rewards_list = []
        self.next_state_tesnsors_list = []
        self.done_list = []
    
    def append(self, state_tensor, action, action_probs, reward, next_state_tensor, done):
        self.state_tesnsors_list.append(state_tensor)
        self.action_list.append(action)
        self.action_probs_list.append(action_probs)
        self.rewards_list.append(reward)
        self.next_state_tesnsors_list.append(next_state_tensor)
        self.done_list.append(done)

    def get_size(self):
        return len(self.state_tesnsors_list)

    def get_shuffled_indices(self):
        return np.random.permutation(self.get_size())

    def clear(self):
        self.state_tesnsors_list.clear()
        self.action_list.clear()
        self.action_probs_list.clear()
        self.rewards_list.clear()
        self.next_state_tesnsors_list.clear()
        self.done_list.clear()
        

class PPO_actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1 = 256, hidden_size2 = 256):
        super(PPO_actor, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1) # first hidden layer
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2) # second hidden layer
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size) # output layer

        self.activation = nn.LeakyReLU() # hidden layers activation function
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

        self.activation = nn.LeakyReLU() # hidden layers activation function

        self.input_size = input_size
        self.output_size = 1
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        out = self.fc3(x) # identity output activation (regression)
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

    def __init__(self, state_size, actions_set_cardinality, env, discount_factor = 0.8, clip_eps = 0.2, actor_lr = 2.5e-4, critic_lr = 1e-3, actor_rep = 15, critic_rep= 5, device = torch.device('cpu')):
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

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.input_size = state_size

        self.output_size = (self.actor.output_size, self.critic.output_size)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def training_step(self, replay_buffer:PPO_Buffer, batch_size = 256, critic_loss_fcn = nn.MSELoss(), epochs = 10):
        """
        Does one epoch of training
        """

        # data extraction and conversion to  ###################################

        buffer_size = replay_buffer.get_size()
        batch_size = min(buffer_size, batch_size)

        # check that buffer_size is multiple of batch_size
        if not buffer_size % batch_size == 0: raise Exception('PPO.PPO_agent.training_epoch() error buffer_size must be multiple of batch_size')

        # lists to tensors
        state_buffer_tensor = torch.stack(replay_buffer.state_tesnsors_list, dim=0).to(device=self.device).to(torch.float) # for each row a step 
        action_buffer = np.array(replay_buffer.action_list) # row vector containing the selected action for every sampled transition
        reward_buffer_tensor = torch.tensor(replay_buffer.rewards_list).to(device=self.device).to(torch.float) # row vector
        next_state_buffer_tensor = torch.stack(replay_buffer.next_state_tesnsors_list, dim=0).to(device=self.device).to(torch.float) # for each row a step  
        action_probs_buffer_tensor = torch.stack(replay_buffer.action_probs_list, dim=0).to(device=self.device).to(torch.float)
        done_buffer_tensor = torch.tensor(replay_buffer.done_list).to(device=self.device) # row vector of zeros and ones 
        done_float_buffer_tensor = done_buffer_tensor.to(torch.float)

        #########################################################################

        # for each epoch
        loss_actor_list, loss_critic_list = [], []
        for _ in range(epochs):
            # get shuffled indeces
            shuffled_indices = replay_buffer.get_shuffled_indices()
            shuffled_indices = shuffled_indices.reshape(-1, batch_size) # iters for epoch x batch_size

            #  1 epoch
            for i in range(shuffled_indices.shape[0]):

                indices = shuffled_indices[i].astype(int)

                state_batch_tensor = state_buffer_tensor[indices]
                action_batch = action_buffer[indices]
                reward_batch_tensor = reward_buffer_tensor[indices]
                next_state_batch_tensor = next_state_buffer_tensor[indices]
                action_probs_batch_tensor = action_probs_buffer_tensor[indices] # pi_old(at|st)
                # done_batch_tensor = done_buffer_tensor[indices]
                done_float_batch_tensor = done_float_buffer_tensor[indices]
                
                states_vals_cr = self.critic(state_batch_tensor).reshape(-1) # column vector to row vector, critic output
                next_states_vals_cr = self.critic(next_state_batch_tensor).reshape(-1) # column vector to row vector, critic output

                
                # TD(0) rewards to go estimates
                rewards_to_go = reward_batch_tensor + self.discount_factor*next_states_vals_cr*(1-done_float_batch_tensor)

                advantages = (rewards_to_go - states_vals_cr).detach()  # TD(0) error used as an approximation of the advantage
                # detach() needed to avoid RuntimeError: Trying to backward through the graph a second time

                # actor parameters update
                for _ in range(self.actor_rep):

                    action_probs_ac = self.actor(state_batch_tensor)[[np.arange(batch_size), action_batch]]

                    importance_sampling_ratio = action_probs_ac / action_probs_batch_tensor

                    loss_actor = -torch.mean(   torch.min(
                                                    advantages * importance_sampling_ratio, 
                                                    advantages * torch.clip(importance_sampling_ratio, min = 1-self.clip_eps, max = 1+self.clip_eps)
                                                    )
                                    )

                    self.actor_opt.zero_grad()
                    loss_actor.backward()
                    self.actor_opt.step()
                    
                    


                # critic parameters update
                for _ in range(self.critic_rep):

                    states_vals_cr = self.critic(state_batch_tensor).reshape(-1) # column vector to row vector
                    next_states_vals_cr = self.critic(next_state_batch_tensor).reshape(-1) # column vector to row vector
                    
                    rewards_to_go = reward_batch_tensor + self.discount_factor*next_states_vals_cr*(1-done_float_batch_tensor)

                    loss_critic = critic_loss_fcn(states_vals_cr, rewards_to_go)
                    
                    self.critic.zero_grad()
                    loss_critic.backward()
                    self.critic_opt.step()

            loss_actor_list.append(loss_actor.detach().cpu())
            loss_critic_list.append(loss_critic.detach().cpu())

        return loss_actor_list, loss_critic_list

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

    

        
            

