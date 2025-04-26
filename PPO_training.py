import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import tqdm
import matplotlib.pyplot as plt

# my modules
import PPO

# Constants and parameters #####################################################
BUFFER_SIZE = 2048
BATCH_SIZE = 256
EPOCHS = 10

MAX_STEPS = 24576

LANES = 3

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features


DISCOUNT_FACTOR = 0.7

ACTOR_LR = 3e-4
CRITIC_LR = 5e-4
CLIP_EPS = 0.2
ACTOR_REP = 10
CRITIC_REP = 15




critc_loss_function =  nn.MSELoss()  # nn.MSELoss() # nn.SmoothL1Loss() 

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'>>> DEVICE =  {device}')


# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)


env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={
                        'observation':{
                            'type': 'Kinematics',
                            "features": ["presence", "x", "y", "vx", "vy"],
                        },
                        
                        'action': {
                            'type': 'DiscreteMetaAction'
                        },
                        'lanes_count': LANES,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        render_mode = 'human'
                        )

print('>>> ENVIRONMENT INITIALIZED')

# initialization of the replay buffer
replay_buffer = PPO.PPO_Buffer()
print('>>> REPLAY BUFFER INITIALIZED')



# Initialize your model
agent = PPO.PPO_agent(state_size=STATE_DIMENSIONALITY, actions_set_cardinality=5, env = env, device = device, discount_factor=DISCOUNT_FACTOR, clip_eps=CLIP_EPS, actor_rep=ACTOR_REP, critic_rep=CRITIC_REP, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop

# data that is going to be collected each step
actor_loss_vals_history = []
critic_loss_vals_history = []

# dara that is going to be collected for each episode
episodes_returns = []
episodes_steps = []



for t in tqdm.trange(MAX_STEPS):
    episode_steps += 1

    # conversion of the state numpy array into a pyTorch tensor
    state_tensor = torch.from_numpy(state).to(device=device)

    # Select the action to be performed by the agent
    action, action_probs = agent.act(state_tensor=state_tensor)

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # next state tensor
    next_state_tensor = torch.from_numpy(next_state).to(device=device)

    state = next_state
    episode_return += reward

    # Store transition in memory
    replay_buffer.append(state_tensor=state_tensor, action=action, action_probs=action_probs[action] ,reward=reward, next_state_tensor=next_state_tensor, done=done)

    ### TRAINING STEP ###
    if replay_buffer.get_size() >= BUFFER_SIZE:
        
        loss_actor_epoch_list, loss_critic_epoch_list = agent.training_step(replay_buffer = replay_buffer, batch_size = BATCH_SIZE, critic_loss_fcn=critc_loss_function, epochs=EPOCHS)
        
        actor_loss_vals_history.extend(loss_actor_epoch_list)
        critic_loss_vals_history.extend(loss_critic_epoch_list)

        # emptying the replay_buffer
        replay_buffer.clear()

    ###########################################
    if done or truncated:
        # print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")
        episodes_returns.append(episode_return)
        episodes_steps.append(episode_steps)
        # Save training information and model parameters
        

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

# saving the trained model for evaluation
torch.save(agent, 'trained_PPO_agent.pt')
print('>>> TRAINED PPO MODEL SAVED')

# saving training data
np.savetxt('training_data/PPO/PPO_training_data_losses.csv', np.transpose([actor_loss_vals_history, critic_loss_vals_history]), header= 'actor_loss, critic_loss' ,delimiter = ',')
np.savetxt('training_data/PPO/PPO_training_data_returns_and_episodes_steps.csv', np.transpose([episodes_returns, episodes_steps]), header='returns,episode_length', delimiter=',')

# plotting data


### actor and critic losses ###
losses_fig, (actor_loss_plot, critic_loss_plot) = plt.subplots(nrows = 1, ncols=2)
losses_fig.suptitle('actor and critic losses')

# actor loss
actor_loss_plot.set_title('actor loss')
actor_loss_plot.plot(actor_loss_vals_history)
actor_loss_plot.set_xlabel('training steps')
# critic loss
critic_loss_plot.set_title('critic loss')
critic_loss_plot.plot(critic_loss_vals_history)
critic_loss_plot.set_xlabel('training steps')

### returns and episode length ###
episode_plot, (returns_plot, episode_steps_plot) = plt.subplots(nrows = 1, ncols=2)
episode_plot.suptitle('returns and episode steps')
# returns
returns_plot.set_title("RETURNS")
returns_plot.plot(episodes_returns, color='blue')
returns_plot.set_xlabel("Episodes")
# episode length
episode_steps_plot.set_title('EPISODE STEPS')
episode_steps_plot.plot(episodes_steps, color='purple')
episode_steps_plot.set_xlabel("Episodes")

plt.show()


print('>>> PROGRAM TERMINATED')