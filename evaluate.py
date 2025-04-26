import gymnasium
import highway_env
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# my modules
import DQN
import PPO

# alternatives: 
# 'DQN'
# 'Duelling_DQN'
# 'PPO'
AGENT_TO_BE_TESTED = 'PPO'

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'>>> DEVICE =  {device}')

env_name = "highway-fast-v0" #"highway-fast-v0" #"highway-v0"
LANES = 3
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

# Initialize your model and load parameters
match(AGENT_TO_BE_TESTED):

    case 'DQN':
        print('>>> DQN model test')
        # model initialization
        DQN_agent = torch.load('trained_DQN_agent.pt', weights_only = False) # weights_only true?
        DQN_agent.to(device=device)
        DQN_agent.eval()

    case 'Duelling_DQN':
        print('>>> DQN model test')
        # model initialization
        Duelling_DQN_agent = torch.load('trained_Duelling_DQN_agent.pt', weights_only = False) 
        Duelling_DQN_agent.to(device=device)
        Duelling_DQN_agent.eval()


    case 'PPO':
        print('>>> PPO model test')
        # model initialization
        PPO_agent = torch.load('trained_PPO_agent.pt', weights_only = False)
        PPO_agent.to(device=device)
        PPO_agent.eval()

    case _:
        raise Exception('AGENT_TO_BE_TESTED is not valid, must be: \'DQN\' or \'Duelling_DQN\' or \'PPO\'')

# evaluation data
steps = []
returns = []
dones = []

# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

EPISODES = 100

while episode <= EPISODES:
    episode_steps += 1
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)

    # Select the action to be performed by the agent
    # Initialize your model and load parameters
    match(AGENT_TO_BE_TESTED):

        case 'DQN':
            # action selection DQN
            action = DQN_agent.act_greedy(state_tensor)

        case 'Duelling_DQN':
            # action selection Duelling DQN
            action = Duelling_DQN_agent.act_greedy(state_tensor)

        case 'PPO':
            # action selection PPO
            action,_ = PPO_agent.act(state_tensor=state_tensor)

        case _:
            raise Exception('AGENT_TO_BE_TESTED is not valid, must be: DQN or Duelling_DQN or PPO')
        

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")
        
        steps.append(episode_steps)
        returns.append(episode_return)
        dones.append(done)

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()


print('>>> EVALUATION ENDED')
print(f'Number of episodes: {EPISODES}')
print(f'Average episode return: {np.mean(returns)}, Average episode number of steps: {np.mean(steps)}, Times crashed: {np.count_nonzero(dones)}, ( percentage: {100*np.count_nonzero(dones)/EPISODES}% )')
print(f'Episode returns std: {np.std(returns)}, Episode number of steps std: {np.std(steps)}')



plt.figure('Plot returns distribution')
plt.hist(returns)
plt.xlabel('Return')

plt.figure('Plot steps number distribution')
plt.hist(steps)
plt.xlabel('Steps')

plt.show()
