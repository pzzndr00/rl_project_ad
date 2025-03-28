# Andrea Pizzolotto Baseline
import numpy as np
import random
import torch
import time
import gymnasium
import highway_env
# import os


def baseline_agent(env, obs, lanes_count):
    
    available_actions = env.unwrapped.get_available_actions() # unwrapped is necessary for some reason
    # print(available_actions)
    ego = obs[1]
    obs_noego = obs[1:]
    lane_w = 1 / lanes_count # the width of the lane is normalized wrt the number of lanes
    eps = 0 # lane tollerance (cars while changing lanes can actually occupy two lanes at the same time) NOT NECESSARY, hard to tune
    th_x = 0.125
    
    # boolean variables used to determine wich lanes are free
    # right current and left are referred to the ego car position
    l_lane_free = False # left
    c_lane_free_b = False # current behind
    c_lane_free_a = False # current ahead  
    r_lane_free = False # right

    # obs vector subdivision for different lanes
    l_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, 2] < - lane_w/2 + eps) and (obs_noego[i, 2] > - 3*lane_w/2 - eps)])
    c_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, 2] >= - lane_w/2 - eps) and (obs_noego[i, 2] <= lane_w/2 + eps)])
    r_cars = np.array([obs_noego[i] for i in range(obs_noego.shape[0]) if (obs_noego[i, 2] > lane_w/2 - eps) and (obs_noego[i, 2] < 3*lane_w/2 + eps)])

    # print(f'cars in the left lane: {l_cars.shape[0]}')
    # print(f'cars in the current lane: {c_cars.shape[0]}')
    # print(f'cars in the right lane: {r_cars.shape[0]}')

    # 
    #os.system('cls' if os.name == 'nt' else 'clear')

    # RIGHT LANE
    # if there are no cars in the right lane and LANE_RIGHT available then
    if not np.any(r_cars) and 2 in available_actions:
        r_lane_free = True # right lane free

    # if there are cars in the right lane but they are far enough and LANE_RIGHT available then
    if np.any(r_cars) and 2 in available_actions:        
        if np.all(abs(r_cars[:,1]) > th_x):
            r_lane_free = True # right lane free

    # CURRENT LANE both ahead and behind
    # if there are no cars in the current lane (current lane always available) then
    if not np.any(c_cars):
        c_lane_free_b = True # current lane free behind
        c_lane_free_a = True # current lane free ahead

    # if there are cars in the current lane but they are far enough then
    # c_cars_a = np.array()
    # c_cars_b = np.array()
    if np.any(c_cars):
        c_cars_a = np.array([c_cars[i] for i in range(c_cars.shape[0]) if c_cars[i, 1] >= 0]) # cars ahead
        if (not np.any(c_cars_a)) or np.all(c_cars_a[:,1] > 1.2*th_x):
            c_lane_free_a = True # current lane free ahead
        
        c_cars_b = np.array([c_cars[i] for i in range(c_cars.shape[0]) if c_cars[i, 1] < 0]) # cars behind
        if (not np.any(c_cars_b)) or np.all(c_cars_b[:,1] < - 0.05 * th_x):
            c_lane_free_b = True # current lane free behind

    # LEFT LANE
    # if there are no cars in the left lane and LANE_LEFT available then
    if not np.any(l_cars) and 0 in available_actions:
        l_lane_free = True # left lane free

    # if there are cars in the left lane but they are far enough and LANE_LEFT available then
    if np.any(l_cars) and 0 in available_actions:
        if all(abs(l_cars[:,1]) > th_x):
            l_lane_free = True # left lane free

    # print(obs)
   
    # print(f'lanes free? r ca cb l {r_lane_free, c_lane_free_a, c_lane_free_b, l_lane_free}')

    # ACTION SELECTION

    # highest priority is to go right if the lane is free
    if r_lane_free:
        return 2 # go right
    
    # if the right lane is not available or is occupied but the current lane is 
    # free go faster in the current lane
    if c_lane_free_a:
        # print('gas')
        if 3 in available_actions:
            return 3    # go faster
        else: return 1  # idle
    
    # if the right and current lane are occupied but the left lane is free
    # go in the left lane and overtake
    if l_lane_free:
        return 0    # go left
    
    # if all the lanes are occupied and ther is no car approaching from behind then
    if c_lane_free_b:
        # if ego can go any slower then
        if 4 in available_actions:
            # print('slow down')
            return 4 # go slower
        # otherwise if the car in front is slower than ego
        else:
            # given the closest car from ahead is slower than ego then
            c_cars_a.reshape(-1, 7)
            closest_ahead = np.argmin(c_cars_a[:, 1])
            if c_cars_a[closest_ahead, 1] < 0.25 * th_x and c_cars_a[closest_ahead, 3] < 0:

                if 2 in available_actions and 0 in available_actions:
                    r_cars.reshape(-1, 7)
                    l_cars.reshape(-1, 7)
                    if np.min(abs(l_cars[:, 1])) < np.min(abs(r_cars[:, 1])):
                        return 2
                    else: return 0
                
                if not 2 in available_actions: return 0
                if not 0 in available_actions: return 3
                
    
    
    # else
    return 1






# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-v0" #"highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={
                        'observation':{
                            'type': 'Kinematics',
                            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                            'vehicles_count': 5,
                        },
                        
                        'action': {
                            'type': 'DiscreteMetaAction'
                        },
                        'lanes_count': 3,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        render_mode = 'human'
                        )


obs, info = env.reset()
done, truncated = False, False
episode = 1
episode_steps = 0
episode_return = 0


while episode <= 10:
    episode_steps += 1

    # action selection
    # action = 4 # always go slower (chi va piano...) 28/30 reward
    # action = env.action_space.sample() # random <= 10
    action = baseline_agent(env = env, obs = obs, lanes_count = 3)

    # step
    obs, reward, done, truncated, info = env.step(action)
    episode_return += reward
    env.render()
    # time.sleep(1)
    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0


env.close()

