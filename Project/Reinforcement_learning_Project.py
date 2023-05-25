import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
from time import sleep
from matplotlib import animation
from tqdm import tqdm


############################ Helper Functions ###############################333

def initial_State(env,title = 'SARSA initial state'):
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



def SARSA(alpha, gamma, epsilon, num_episodes, env, total_epochs, cum_rewards,q_table):
    for episode in tqdm(range(1, num_episodes+1)):
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
        cum_rewards[episode-1] = cum_reward/epoch

        # if episode % 100 == 0:
        #     clear_output(wait=True)
        #     print(f"Episode #: {episode}")

    print("\n")
    print("===Training completed.===\n")

def Q_learning(alpha, gamma, epsilon, num_episodes, env, total_epochs, cum_rewards,q_table):
    for episode in tqdm(range(1, num_episodes+1)):
        # Reset environment
        state, info = env.reset()
        epoch = 0 
        num_failed_dropoffs = 0
        done = False
        cum_reward = 0

        while not done: #Every step in the episode
            
            action = epsilon_greedy(env, state, q_table, epsilon)
    
            next_state, reward, done, _ , info = env.step(action) 

            cum_reward += reward
            
            old_q_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
            
            q_table[state, action] = new_q_value
            
            if reward == -10:
                num_failed_dropoffs += 1

            state = next_state
            epoch += 1
            
        total_epochs[episode-1] = epoch
        cum_rewards[episode-1] = cum_reward/epoch

    print("\n")
    print("===Training completed.===\n")

rng = np.random.default_rng()
def epsilon_greedy(env, state, q_table, epsilon):
    if rng.random() < epsilon:
    # if random.uniform(0, 1) < epsilon:
        "Basic exploration [~0.47m]"
        action = env.action_space.sample() # Sample random action (exploration)
        
    else:      
        
        "Basic exploitation [~47s]"
        action = np.argmax(q_table[state]) # Select best known action (exploitation)
    return action




''' 
First we will perform training and testing for SARSA agent
'''
############### Setup Environment ############################

env_SARSA = gym.make("Taxi-v3", render_mode="rgb_array").env
env_Qlearning = gym.make("Taxi-v3", render_mode="rgb_array").env

initial_State(env_SARSA, "SARSA Initial State")
initial_State(env_Qlearning, "QLearning Initial State")



####################### Training the Agent #############################


q_table_SARSA= np.zeros([env_SARSA.observation_space.n, env_SARSA.action_space.n])
q_table_QLearning= np.zeros([env_Qlearning.observation_space.n, env_SARSA.action_space.n])

# Hyperparameters
alpha = 0.05  # Learning rate 
gamma = 0.95  # Discount rate
epsilon = 0.1  # Exploration rate
num_episodes = 10000  # Number of episodes


# Output for plots
cum_rewards_SARSA = np.zeros([num_episodes])
total_epochs_SARSA = np.zeros([num_episodes])
cum_rewards_QLearning = np.zeros([num_episodes])
total_epochs_QLearning = np.zeros([num_episodes])


print(f'\n\n\n\t\t\t Let the SARSA training begin \n\n\n ')
SARSA(alpha, gamma, epsilon, num_episodes, env_SARSA, total_epochs_SARSA, 
      cum_rewards_SARSA ,q_table_SARSA)


print(f'\n\n\n\t\t\t Let the Q learning training begin \n\n\n ')
SARSA(alpha, gamma, epsilon, num_episodes, env_Qlearning, total_epochs_QLearning, 
      cum_rewards_QLearning ,q_table_QLearning)


####### Plotting the results
# Plot reward convergence Seperate
plt.title("Average Reward Per Episode for SARSA")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.plot(cum_rewards_SARSA, label = f'SARSA Average reward per episode', color = 'red')
plt.legend()
plt.xlim([0, 500])
plt.show()
plt.title("Average Reward Per Episode for Q-learning")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.plot(cum_rewards_QLearning, label = f'QLearning Average reward per episode ')
plt.legend()
plt.xlim([0, 500])
plt.show()


# Plot reward convergence together
plt.title("Average Reward Per Episode ")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.plot(cum_rewards_SARSA, label = f'SARSA Average reward per episode' , color = 'red')
plt.plot(cum_rewards_QLearning, label = f'QLearning Average reward per episode ')
plt.legend()
plt.xlim([0, 500])
plt.show()


# Plot epoch convergence seperate
plt.title("# Steps Per Episode for SARSA")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs_SARSA, label = 'SARSA total steps in episode ',  color = 'red')
plt.legend()
plt.xlim([0, 500])
plt.show()
plt.title("# Steps Per Episode for SARSA")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs_QLearning, label = 'QLearning total steps in episode')
plt.legend()
plt.xlim([0, 500])
plt.show()


# Plot epoch convergence
plt.title("# epochs per episode")
plt.xlabel("Episode")
plt.ylabel("# epochs")
plt.plot(total_epochs_SARSA, label = 'SARSA total steps in episode',  color = 'red')
plt.plot(total_epochs_QLearning, label = 'QLearning total steps in episode ')
plt.legend()
plt.xlim([0, 500])
plt.show()

##### The test policy below has some error ###########

# """Test policy performance after training"""
# print(f'The Q table for SARSA is \n {q_table_SARSA}')
# num_epochs = 0
# total_failed_deliveries = 0
# num_episodes = 200
# experience_buffer = []
# store_gif = True

# total_reward =0
# for episode in tqdm(range(1, num_episodes+1)):
#     # Initialize experience buffer

#     my_env = env_SARSA.reset()
   
#     state = my_env[0]
#     epoch = 1 
#     num_failed_deliveries =0
#     cum_reward = 0
#     done = False

#     i = 0

#     while not done:
#         # action = epsilon_greedy(env, state, q_table, epsilon)
#         action = np.argmax(q_table_SARSA[state]) #what if we take epsilon greedy herer
#         state, reward, done, _, _ = env_SARSA.step(action)
#         cum_reward += reward

#         if reward == -10:
#             num_failed_deliveries += 1
        
#         # Store rendered frame in animation dictionary
#         i +=1
#         if i==1000:
#             break
#         # if episode == num_episodes:
#         # #     experience_buffer.append({
#         # #         'frame': env_SARSA.render(),
#         # #         'episode': episode,
#         # #         'epoch': epoch,
#         # #         'state': state,
#         # #         'action': action,
#         # #         'reward': cum_reward
#         # #         }
#         # # )
       

#         epoch += 1
#     total_reward += cum_reward
#     total_failed_deliveries += num_failed_deliveries
#     num_epochs += epoch

# if store_gif:
#     store_episode_as_gif(experience_buffer, filename="SARSA.gif")

# print("\n") 
# print("This is the results for SARSA")
# print(f"Test results after {num_episodes} episodes:")
# print(f"Mean # epochs per episode: {num_epochs / num_episodes}")
# print(f"The mean award per episodes are  {total_reward/num_episodes}")
# print(f"Mean # failed drop-offs per episode: {total_failed_deliveries / num_episodes}")

# print(f'\n\n\n\n\n\n')

# print(f'The Q table for Q_Learning is \n {q_table_QLearning}')
# num_epochs = 0
# total_failed_deliveries = 0
# num_episodes = 1
# experience_buffer = []
# store_gif = True

# total_reward =0
# for episode in tqdm(range(1, num_episodes+1)):
#     # Initialize experience buffer

#     my_env = env_SARSA.reset()
   
#     state = my_env[0]
#     epoch = 1 
#     num_failed_deliveries =0
#     cum_reward = 0
#     done = False
#     i = 0
#     while not done:
#         # action = epsilon_greedy(env, state, q_table, epsilon)
#         action = np.argmax(q_table_QLearning[state]) #what if we take epsilon greedy herer
#         state, reward, done, _, _ = env_Qlearning.step(action)
#         cum_reward += reward

#         i +=1
#         if i==1000:
#             break
#         if reward == -10:
#             num_failed_deliveries += 1

#         epoch += 1
#     total_reward += cum_reward
#     total_failed_deliveries += num_failed_deliveries
#     num_epochs += epoch


# print("\n") 
# print("This is the results for Q-Learner")
# print(f"Test results after {num_episodes} episodes:")
# print(f"Mean # epochs per episode: {num_epochs / num_episodes}")
# print(f"The mean award per episodes are  {total_reward/num_episodes}")
# print(f"Mean # failed drop-offs per episode: {total_failed_deliveries / num_episodes}")