"""
Experiements to run
 - use a bottlneck layer for identifying how many times the network has seen this state to calculate the regret for exploration (instead of epsilon greedy)
 - use a NN that given the S,A predicts S'.  Use this to calculate the S' that is least like the current state and use that as the exploration
 - or like diffussion, predict delta S
 - use a second NN to predict the reward given the S,A and use the difference in predicted Q to the first NN to estimate regret for exploration
"""

import pickle
from plottool import plot_many
import numpy as np
from dqn import DQN, q_learning
import gym
import os

env = gym.make('CartPole-v1')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
episodes = 500
trials = 20

save_file = "simple.pickle"
results = []
labels = []

experiments = (
    ("64 resnet", lambda:q_learning(env=env, model=DQN(state_dim=n_state, action_dim=n_action) , episodes=episodes, optimism=0.9)),
    ("64 resnet + rand", lambda:q_learning(env=env, model=DQN(state_dim=n_state, action_dim=n_action, rand=0.001) , episodes=episodes, optimism=0.9)),
)

if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        labels, results = pickle.load(f)

if True:
    for label, worker in experiments[len(labels):]:
        print(end=f"{label} = ")
        samples = []
        for _ in range(trials):
            samples.append( worker() )
            print(end=f"{len(samples[-1]):3} ")
            if _ % 5:
                print(end="| ")
        print()
        results.append(samples)
        labels.append(label)
        # save the data
        with open(save_file, 'wb') as f:
            pickle.dump([labels, results], f)

averages = []
for result in results:
    average = np.array([ x + [x[-1]]*(episodes-len(x)) for x in result ]).mean(0).tolist()
    averages.append(average)

plot_many(averages, titles=labels)

