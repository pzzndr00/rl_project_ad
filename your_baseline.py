# Andrea Pizzolotto Baseline
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


import gymnasium
import highway_env

def baseline_agent(env, obs, lanes_count):
    """
    Baseline agent (algorithmic) that will be used as a comparison for the trained RL agent

    Parameters
    ---------

    env: the gymnasium environment representing a highway_env environment (either "highway-v0" or "highway-fast-v0")
    obs: the current state observation (normalized distances)
    lanes_count: the number of lanes for the environment

    Returns
    -------

    action: the selected action from one of the following {'LANE_LEFT': 0, 'IDLE':1, 'LANE_RIGHT':2, 'FASTER':3, 'SLOWER':4}

    """
    available_actions = env.unwrapped.get_available_actions() # unwrapped is necessary to access the highway_env object

    # dictionaries defined below help with the interpretability of the code

    # I didn't find a better way of getting a dictionary of actions where the lables are the actions' names
    actions_lable_map = highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL
    actions = {actions_lable_map[0]: 0,
               actions_lable_map[1]: 1,
               actions_lable_map[2]: 2,
               actions_lable_map[3]: 3,
               actions_lable_map[4]: 4,
        }
    
    
    # similarly for features ...
    features_lables = highway_env.envs.common.observation.KinematicObservation.FEATURES

    features = {features_lables[0]: 0,
                features_lables[1]: 1,
                features_lables[2]: 2,
                features_lables[3]: 3,
                features_lables[4]: 4,
        }
    
    ego = obs[1]
    obs_noego = obs[1:]
    lane_w = 1 / lanes_count # the width of the lane is normalized wrt the number of lanes
    eps = 0 # lane tollerance (cars while changing lanes can actually occupy two lanes at the same time) NOT NECESSARY, hard to tune
    th_x = 0.125
    
    # boolean variables used to determine wich lanes are free
    # right current and left are referred with respect to the ego car position, the current lane divided in ahead and behind
    l_lane_free = False # left
    c_lane_free_b = False # current behind
    c_lane_free_a = False # current ahead  
    r_lane_free = False # right

    # obs vector subdivision for different lanes
    l_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, features['y']] < - lane_w/2 + eps) and (obs_noego[i, features['y']] > - 3*lane_w/2 - eps)])
    c_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, features['y']] >= - lane_w/2 - eps) and (obs_noego[i,features['y']] <= lane_w/2 + eps)])
    r_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, features['y']] > lane_w/2 - eps) and (obs_noego[i, features['y']] < 3*lane_w/2 + eps)])

    # RIGHT LANE ###############################################################
    # if there are no cars in the right lane and LANE_RIGHT available then
    if not np.any(r_cars) and actions['LANE_RIGHT'] in available_actions:
        r_lane_free = True # right lane free

    # if there are cars in the right lane but they are far enough and LANE_RIGHT available then
    if np.any(r_cars) and actions['LANE_RIGHT'] in available_actions:        
        if np.all(abs(r_cars[:,features['x']]) > th_x):
            r_lane_free = True # right lane free
    ############################################################################

    # CURRENT LANE both ahead and behind #######################################
    # if there are no cars in the current lane (current lane always available) then
    if not np.any(c_cars):
        c_lane_free_b = True # current lane free behind
        c_lane_free_a = True # current lane free ahead

    # if there are cars in the current lane but they are far enough then
    if np.any(c_cars):
        c_cars_a = np.array([c_cars[i] for i in range(c_cars.shape[0]) if c_cars[i, features['x']] >= 0]) # cars ahead
        if (not np.any(c_cars_a)) or np.all(c_cars_a[:, features['x']] > 1.1*th_x):
            c_lane_free_a = True # current lane free ahead
        
        c_cars_b = np.array([c_cars[i] for i in range(c_cars.shape[0]) if c_cars[i, features['x']] < 0]) # cars behind
        if (not np.any(c_cars_b)) or np.all(c_cars_b[:, features['x']] < - 0.05 * th_x):
            c_lane_free_b = True # current lane free behind
    ############################################################################


    # LEFT LANE ################################################################
    # if there are no cars in the left lane and LANE_LEFT available then
    if not np.any(l_cars) and actions['LANE_LEFT'] in available_actions:
        l_lane_free = True # left lane free

    # if there are cars in the left lane but they are far enough and LANE_LEFT available then
    if np.any(l_cars) and actions['LANE_LEFT'] in available_actions:
        if all(abs(l_cars[:, features['x']]) > th_x):
            l_lane_free = True # left lane free
    ############################################################################

    # ACTION SELECTION

    # highest priority is to go right if the lane is free
    if r_lane_free:
        return actions['LANE_RIGHT'] # go right
    
    # if the right lane is not available or is occupied but the current lane is 
    # free go faster in the current lane if possible otherwise idle
    if c_lane_free_a:
        if actions['FASTER'] in available_actions:
            return actions['FASTER']  # go faster
        else: return actions['IDLE']  # idle
    
    # if the right and current lane are occupied but the left lane is free
    # go in the left lane and overtake
    if l_lane_free:
        return actions['LANE_LEFT']    # go left
    
    # if all the lanes are occupied and ther is no car approaching from behind then
    if c_lane_free_b: # actually never happens to have a car approaching from behind
        # if ego can go slower then
        if actions['SLOWER'] in available_actions:
            return actions['SLOWER'] # go slower
    
    # if ego can't go any slower then
    if not actions['SLOWER'] in available_actions:
        c_cars_a.reshape(-1, 5)
        closest_ahead = np.argmin(c_cars_a[:, features['x']]) # extracting information about the closest car ahead
        # if the car ahead is very close and it's going slower than ego (that if in this section is already at minimum speed) then
        # it's necessary to try to avoid the collision 
        if c_cars_a[closest_ahead, features['x']] < 0.35 * th_x and c_cars_a[closest_ahead, features['vx']] < 0:
            # if both LANE_RIGHT and LANE_LEFT available then chose the lane which cars in it are further
            if actions['LANE_RIGHT'] in available_actions and actions['LANE_LEFT'] in available_actions:
                r_cars.reshape(-1, 5)
                l_cars.reshape(-1, 5)
                if np.min(abs(l_cars[:, features['x']])) < np.min(abs(r_cars[:, features['x']])):
                    return actions['LANE_RIGHT']
                else: return actions['LANE_LEFT']

            # if LANE_RIGHT not available then just escape going in the left lane
            if not actions['LANE_RIGHT'] in available_actions:
                # print('going left because right not available')
                return actions['LANE_LEFT']                
            # if LANE_LEFT not available then just escape going in the right lane
            if not actions['LANE_LEFT'] in available_actions:
                # print('going right because left not available')
                return actions['LANE_RIGHT']

    # else
    return actions['IDLE']


def cheat_baseline():
    """
    Cheating baseline for highway_env (either "highway-v0" or "highway-fast-v0"), always returns 4 that corresponds to the action 'SLOWER'
    Even if is very simple it 'breakes' the game because it never really happens to find a car approaching from behind causing a collision
    this means that the only possible cause for collision is when a car in the same lane goes slower than the minimum possible speed of the
    ego car (the agent). This in the end results in a overall higher episode return average than the more greedy (and complex) 'baseline_policy' 
    """
    return 4



# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  #"highway-v0" #"highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

LANES = 3
EPISODES = 100

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

# evaluation data
steps = []
returns = []
dones = []

obs, info = env.reset()
done, truncated = False, False
episode = 1
episode_steps = 0
episode_return = 0

while episode <= EPISODES:
    episode_steps += 1

    # baseline action selection
    action = baseline_agent(env = env, obs = obs, lanes_count = LANES)

    # step
    obs, reward, done, truncated, info = env.step(action)

    episode_return += reward
    env.render()

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        steps.append(episode_steps)
        returns.append(episode_return)
        dones.append(done)

        env.reset()
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