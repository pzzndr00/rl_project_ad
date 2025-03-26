# Andrea Pizzolotto Baseline
import numpy as np
import random
import torch

import gymnasium
import highway_env


# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={
                        'observation':{
                            'type': 'OccupancyGrid'
                        },

                        'action': {
                            'type': 'DiscreteMetaAction'
                        },

                        'duration': 40, "vehicles_count": 50},
                        render_mode = 'human'
                        )


env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= 10:
    episode_steps += 1

    # action selection
    action = 4 # always go slower (chi va piano...) 28/30 reward
    # action = env.action_space.sample() # random <= 10

    # step
    _, reward, done, truncated, _ = env.step(action)
    episode_return += reward
    env.render()

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0


env.close()