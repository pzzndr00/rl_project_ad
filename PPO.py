import torch
import torch.nn as nn
import random
import numpy as np

import memoryBuffer as mb

# class trj_set(object):
#     def __init__(self):
#         super(trajectory, self).__init__()

#         self.states = []        # S
#         self.actions = []       # A
#         self.rewards = []       # R
#         self.next_states = []   # S' redoundant but helps in training
        
#         self.done = []

#     def append_transition(self, state, action, reward, next_state, probs, done):

#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.next_states.append(next_state)

#         self.probs.append(probs)

#         self.done.append(done)

#     def __len__(self):
#         return len(self.states)




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


class PPO_agent(nn.Module):

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

    def forward(self, x):
        return self.actor(x), self.critic(x)


    def training_step(self, batch_size:int,  replay_buffer:mb.memory, critic_loss_function = nn.MSELoss()):

        batch_size = min(batch_size, len(replay_buffer))

        # batch sampling and conversion to tensors #############################
        batch = replay_buffer.sample(batch_size = batch_size)

        batch_tr = mb.Transition(*zip(*batch))


        state_batch_tensor = torch.cat(batch_zip.state).reshape((-1, self.input_size)).to(device=device) # for each row a step       
        action_batch = np.array(batch_zip.action) # row vector
        reward_batch_tensor = torch.cat(batch_zip.reward).reshape((-1,1)).to(device=device) # column vector
        next_state_batch_tensor = torch.cat(batch_zip.next_state).reshape((-1, self.input_size)).to(device=device) # for each row a step  

        done_batch_tensor = torch.cat(batch_zip.done.bool().int()).reshape(-1,1).to(device=device) # column vector of zeros and ones 

        states_vals_cr = self.critic(state_batch_tensor) # column vector (?)
        next_states_vals_cr = self.critic(next_state_batch_tensor) # column vector (?)

        initial_action_probs_ac = self.actor(state_batch_tensor)[action_batch] # row vector with selected action pi_old(at|st)

        with torch.no_grad():
            
            rewards_to_go = reward_batch_tensor + self.discount_factor*next_states_vals_cr*(1-done_batch_tensor)

        td_error = rewards_to_go - states_vals_cr
        
        # actor parameters oprtimization

        for _ in range(self.actor_rep):
            action_probs_ac = self.actor(state_batch_tensor)[action_batch]

            importance_sampling_ratio = action_probs_ac / initial_action_probs_ac
            
            loss_actor = torch.minimum()

            ## si riprende da qua ###########################################################################################


        # critic parameters optimization
        for _ in range(self.critic_rep):
            raise NotImplementedError


        raise NotImplementedError

    def act(self, state_tensor:torch.tensor):
        """

        Function that choses an action according to the output probabilities given by the actor for the given state
        PARAMETERS
        ----------
        state_tensor: a pytorch tensor containing the vectorized state of highway-env

        RETURNS
        -------
        action: the selected action
        pi_given_s: the output of the ctitic given the current state 

        """
        probs = self.actor(state_tensor).cpu().detach().numpy()
        action = random.choices(probs.shape[0], probs)

        return action, pi_given_s

    

        
            

