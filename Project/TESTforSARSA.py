import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation


def initial_State(env, title = 'Initial Sate of SARSA env'):
    state, _ = env.reset()

    # Print dimensions of state and action space
    print("State space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))

    # Sample random action
    action = env.action_space.sample(env.action_mask(state))
    next_state, reward, done, _, _ = env.step(action)

    # Print output
    print("State: {}".format(state))
    print("Action: {}".format(action))
    print("Action mask: {}".format(env.action_mask(state)))
    print("Reward: {}".format(reward))

    # Render and plot an environment frame
    frame = env.render()
    plt.imshow(frame)
    plt.title(title)
    plt.axis("off")
    plt.show()





def SARSA(alpha, gamma, epsilon, num_episodes, env, total_epochs, cum_rewards):
    for episode in range(1, num_episodes+1):
        # Reset environment
        state, info = env.reset()
        epoch = 0 
        num_failed_dropoffs = 0
        done = False
        cum_reward = 0
        action = epsilon_greedy(env, state, q_table, epsilon) #This gives us A from S using Q
        while not done: #For each step of the episode
            
            next_state, reward, done, _ , info = env.step(action) #Take action A observe R, S'

            next_action = epsilon_greedy(env, next_state, q_table, epsilon) #choose A' from S' using policy derived from Q
            

            cum_reward += reward

            old_q_value = q_table[state, action]
            new_q_value = q_table[next_state, next_action]
            
            new_q_value =  (1- alpha) * old_q_value + alpha * (reward + gamma * new_q_value)
            
            q_table[state, action] = new_q_value
            
            if reward == -10:
                num_failed_dropoffs += 1

            state = next_state
            action = next_action
            epoch += 1
            
        total_epochs[episode-1] = epoch
        cum_rewards[episode-1] = cum_reward

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode #: {episode}")

    print("\n")
    print("===Training completed.===\n")

def epsilon_greedy(env, state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        "Basic exploration [~0.47m]"
        action = env.action_space.sample() # Sample random action (exploration)
        
    else:      
        
        "Basic exploitation [~47s]"
        action = np.argmax(q_table[state]) # Select best known action (exploitation)
    return action



def run_animation(experience_buffer):
    """Function to run animation"""
    time_lag = 0.05  # Delay (in s) between frames
    for experience in experience_buffer:
        # Plot frame
        clear_output(wait=True)
        plt.imshow(experience['frame'])
        plt.axis('off')
        plt.show()

        # Print console output
        print(f"Episode: {experience['episode']}/{experience_buffer[-1]['episode']}")
        print(f"Epoch: {experience['epoch']}/{experience_buffer[-1]['epoch']}")
        print(f"State: {experience['state']}")
        print(f"Action: {experience['action']}")
        print(f"Reward: {experience['reward']}")
        # Pauze animation
        sleep(time_lag)

def store_episode_as_gif(experience_buffer, path='./', filename='animation.gif'):
    """Store episode as gif animation"""
    fps = 5   # Set framew per seconds
    dpi = 300  # Set dots per inch
    interval = 50  # Interval between frames (in ms)

    # Retrieve frames from experience buffer
    frames = []
    for experience in experience_buffer:
        frames.append(experience['frame'])

    # Fix frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    # Generate animation
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)

    # Save output as gif
    anim.save(path + filename, writer='imagemagick', fps=fps)


def plots(cum_rewards, total_epochs):
    # Plot reward convergence
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cum_rewards)
    plt.show()

    # Plot epoch convergence
    plt.title("# epochs per episode")
    plt.xlabel("Episode")
    plt.ylabel("# epochs")
    plt.plot(total_epochs)
    plt.show()

env = gym.make("Taxi-v3", render_mode="rgb_array").env
initial_State(env)


"""Training the agent"""
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.05  # Learning rate 
gamma = 0.95  # Discount rate
epsilon = 0.1  # Exploration rate
num_episodes = 5000  # Number of episodes
        


# Output for plots
cum_rewards = np.zeros([num_episodes])
total_epochs = np.zeros([num_episodes])

SARSA(alpha, gamma, epsilon, num_episodes, env,total_epochs, cum_rewards )





# plots(cum_rewards, total_epochs)
"""Test policy performance after training"""

num_epochs = 0
total_failed_deliveries = 0
num_episodes = 200
experience_buffer = []
store_gif = True

total_reward =0
for episode in range(1, num_episodes+1):
    # Initialize experience buffer

    my_env = env.reset()
    print(f'this is my env {my_env}')
    state = my_env[0]
    epoch = 1 
    num_failed_deliveries =0
    cum_reward = 0
    done = False
    i = 0
    while not done:
        # action = epsilon_greedy(env, state, q_table, epsilon)
        action = np.argmax(q_table[state]) #what if we take epsilon greedy herer
        state, reward, done, _, _ = env.step(action)
        cum_reward += reward

        i +=1
        if i==1000:
            break
        if reward == -10:
            num_failed_deliveries += 1

        # Store rendered frame in animation dictionary
        if episode == num_episodes:
            experience_buffer.append({
                'frame': env.render(),
                'episode': episode,
                'epoch': epoch,
                'state': state,
                'action': action,
                'reward': cum_reward
                }
        )

        epoch += 1
    total_reward += cum_reward
    total_failed_deliveries += num_failed_deliveries
    num_epochs += epoch

if store_gif:
    store_episode_as_gif(experience_buffer, filename="SARSA.gif")

# Run animation and print output
run_animation(experience_buffer)

# Print final results
print("This is the results for SARSA learning")
print(f"Test results after {200} episodes:")
print(f"Mean # epochs per episode: {num_epochs / num_episodes}")
print(f"The mean award per episodes are  {total_reward/num_episodes}")
print(f"Mean # failed drop-offs per episode: {total_failed_deliveries / num_episodes}")