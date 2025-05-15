import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import tqdm
import matplotlib.pyplot as plt
import os

# my modules
import PPO

# Constants and parameters #####################################################
BUFFER_SIZE = 2048 # 1024 - 2048 # must be a multiple of BATCH_SIZE
BATCH_SIZE = 256 # 64 - 128 - 256
EPOCHS = 10

MAX_STEPS = 24576 # better if a multiple of BUFFER_SIZE, extra samples will be thrown away
NUMBER_OF_TRAINING_STEPS = int(MAX_STEPS/BUFFER_SIZE)
print(f'The algorithm will perform {NUMBER_OF_TRAINING_STEPS} training steps')
LANES = 3

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

DISCOUNT_FACTOR = 0.85

ACTOR_LR = 2e-4
CRITIC_LR = 5e-4

# Entropy linear decay parameters
LIN_DEC = True

ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END = 0.01
DECAY_END_PERCENTAGE = 0.5 # indicates at which portion of training the entropy coefficient gets to its final value

entropy_coef = 0 # 0.01 # 0 = no entropy, standard clip loss # (if LIN_DEC = True entropy_coef is computed following the linear decay and this value is ignored)

CLIP_EPS = 0.1
ACTOR_REP = 15
CRITIC_REP = 5

CLIP_GRAD = False

critc_loss_function =  nn.MSELoss()  # nn.MSELoss() # nn.SmoothL1Loss() 

ADV_EST = 'GAE'

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'>>> DEVICE =  {device}')


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


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
                        #render_mode = 'human'
                        )
env.unwrapped.config['high_speed_reward'] = 0.7

print('>>> ENVIRONMENT INITIALIZED')

# initialization of the replay buffer
PPO_buffer = PPO.PPO_Buffer()
print('>>> REPLAY BUFFER INITIALIZED')


# Initialize your model
agent = PPO.PPO_agent(state_size=STATE_DIMENSIONALITY, actions_set_cardinality=5, device = device, discount_factor=DISCOUNT_FACTOR, clip_eps=CLIP_EPS, actor_rep=ACTOR_REP, critic_rep=CRITIC_REP, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

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

# data that is going to be collected for each episode
episodes_returns = []
episodes_steps = []

training_steps = 0

for t in tqdm.trange(MAX_STEPS):
    episode_steps += 1

    # conversion of the state numpy array into a pyTorch tensor
    state_tensor = torch.from_numpy(state).to(device=device)

    # Select the action to be performed by the agent
    action, action_log_prob = agent.act(state_tensor=state_tensor)

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # next state tensor
    next_state_tensor = torch.from_numpy(next_state).to(device=device)

    # Store transition in memory
    PPO_buffer.append(state_tensor=state_tensor, action=action, action_log_prob=action_log_prob ,reward=reward, next_state_tensor=next_state_tensor, done=done)

    state = next_state
    episode_return += reward    

    ### TRAINING STEP ###
    if PPO_buffer.get_size() >= BUFFER_SIZE:
        
        # entropy coefficient linear decay
        if LIN_DEC:
            if entropy_coef > ENTROPY_COEF_END:
                entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * training_steps / int(NUMBER_OF_TRAINING_STEPS*DECAY_END_PERCENTAGE)
                if entropy_coef < ENTROPY_COEF_END: entropy_coef = ENTROPY_COEF_END

        # print(f'entropy coefficient = {entropy_coef}')

        loss_actor_epoch_list, loss_critic_epoch_list = agent.training_step(PPO_buffer=PPO_buffer, batch_size=BATCH_SIZE, critic_loss_fcn=critc_loss_function, epochs=EPOCHS, entropy_coef=entropy_coef, clip_gradients=CLIP_GRAD, adv_est=ADV_EST, normalize_advantages=False)

        actor_loss_vals_history.extend(loss_actor_epoch_list)
        critic_loss_vals_history.extend(loss_critic_epoch_list)

        # emptying the PPO_buffer
        PPO_buffer.clear()

        training_steps += 1

    ###########################################
    if done or truncated:
        # print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")
        episodes_returns.append(episode_return)
        episodes_steps.append(episode_steps)

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

# saving the trained model for evaluation
if not os.path.exists('trained_models_weights'): os.makedirs('trained_models_weights')
torch.save(agent.state_dict(), 'trained_models_weights/trained_PPO_agent.pt')
print('>>> TRAINED PPO MODEL SAVED')

# saving training data

if not os.path.exists('training_data/PPO'): os.makedirs('training_data/PPO')

np.savetxt('training_data/PPO/PPO_training_data_actor_loss.csv', np.transpose(actor_loss_vals_history), header= 'actor_loss' ,delimiter = ',')
np.savetxt('training_data/PPO/PPO_training_data_critic_loss.csv', np.transpose(critic_loss_vals_history), header='critic_loss')
np.savetxt('training_data/PPO/PPO_training_data_returns_and_episodes_steps.csv', np.transpose([episodes_returns, episodes_steps]), header='returns,episode_length', delimiter=',')

# plotting training data

os.system('python data_plotter.py --input PPO')

print('>>> PROGRAM TERMINATED')