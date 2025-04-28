import gymnasium
import highway_env
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 25

# Remember to save what you will need for the plots
steps = []
returns = []
dones = []

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= EPISODES:
    episode_steps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    _, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

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

print('>>> MANUAL CONTROL ENDED')
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
