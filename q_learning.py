''' Author's Name: Gelana Tostaeva
    Author's Last Modified Date: April 25, 2020
    Modified by: Group 3 on December 12, 2021
    Revision History by Group 3: Added file comments and headers
    Group 3 members:    Joanna Lu (Section 1) 300916162
                        Kryselle Celine Matienzo(Section 1) 301026753
                        Wen Sophie Xu(Section 1) 301098127
                        Kangle Jiang(Section 2) 300952654
    Program Description: Q-Learning and the Taxi Problem. We will apply the Q-learning algorithm
    to help the taxi agent do its job. The goal of this program is our agent which is the taxi driver
    should be able to deliver a passenger to their destination.
    Revision History: Added file comments and headers
    project name: Q-learning with numpy and OpenAI Taxi-v3
'''
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
# Fixing seed for reproducibility
np.random.seed(0)
# Loading and rendering the gym environment
env = gym.make("Taxi-v3").env
# Getting the state space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
# STEP 1 - Initializing the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))
# Setting the hyperparameters
alpha = 0.7                 # learning rate
discount_factor = 0.618     # discounting rate / discount reward
epsilon = 1                 # Exploration rate, start by exploration, then Exploitation
max_epsilon = 1             # Exploration probability at start
min_epsilon = 0.01          # Minimum exploration probability
decay = 0.01                # Exponential decay rate for exploration probability
train_episodes = 2000
max_steps = 100     # Max steps per episode
# Training the agent
# Creating lists to keep track of reward and epsilon values
training_rewards = []
epsilons = []
for episode in range(train_episodes):
    # Resetting the environment each time as per requirement
    state = env.reset()
    # Starting the tracker for the rewards
    total_training_rewards = 0
    print_steps = False
    if episode in [1800]:
        print_steps = True
        print(episode)
        env.render()
    else:
        print_steps = False
    for step in range(max_steps):
        # Choosing an action given the states based on a random number
        # exploration exploitation trade-off
        exp_exp_tradeoff = random.uniform(0, 1)
        # STEP 2: SECOND option for choosing the initial action - exploit
        # If the random number is larger than epsilon: employing exploitation
        # and selecting best action
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state, :])
        # STEP 2: FIRST option for choosing the initial action - explore
        # Otherwise, employing exploration: choosing a random action
        else:
            action = env.action_space.sample()

        # STEPs 3 & 4: performing the action and getting the reward
        # Taking the action and getting the reward and outcome state
        # We receive +20 points for a successful drop-off and
        # lose 1 point for every time-step it takes.
        # There is also a 10 point penalty for illegal pick-up and drop-off actions.
        new_state, reward, done, info = env.step(action)
        if print_steps:
            env.render()
            print(state,action)
        # STEP 5: update the Q-table
        # Updating the Q-table using the Bellman equation
        Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        # Increasing our total reward and updating the state
        total_training_rewards += reward
        state = new_state

        # Ending the episode
        if done:
            if print_steps:
                print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                print ("Total steps it takes is", step+1)
            break
    # Cutting down on exploration by reducing the epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    # Adding the total reward and reduced epsilon values
    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)
print("Training score over time: " + str(sum(training_rewards) / train_episodes))
# Visualizing results and total reward over all episodes
x = range(train_episodes)
plt.plot(x, training_rewards)
plt.xlabel('Episode')
plt.ylabel('Training total reward')
plt.title('Total rewards over all episodes in training')
plt.show()


# Visualizing the epsilons over all episodes
plt.plot(epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title("Epsilon for episode")
plt.show()
