import gym
import numpy as np
from src import MultiArmedBandit, QLearning
import matplotlib.pyplot as plt

r1 = np.zeros(100)
r2 = np.zeros(100)
r3 = np.zeros(100)
env = gym.make('FrozenLake-v0')
agent1 = QLearning(epsilon=.01)
agent2 = QLearning(epsilon=.5)
agent3 = QLearning(epsilon=.5, adaptive=True)

for i in range(10):
    action_values, rewards = agent1.fit(env, 100000)
    r1 += rewards
    action_values, rewards = agent2.fit(env, 100000)
    r2 += rewards
    action_values, rewards = agent3.fit(env, 100000)
    r3 += rewards


plt.plot(r1/10, label="Epsilon = .01")
plt.plot(r2/10, label="Epsilon = .5")
plt.plot(r3/10, label="Epsilon (adaptive) = .5")

plt.legend()
plt.show()
