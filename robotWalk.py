import gym

# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
env = gym.make('BipedalWalker-v3')
for i_episode in range(100):
   observation = env.reset()
   for t in range(10000):
      env.render()
      print(observation)
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      if done:
         print("{} timesteps taken for the episode".format(t+1))
         break
