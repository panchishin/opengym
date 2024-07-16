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

env = gym.make('CartPole-v1')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
episodes = 500
trials = 20

save_file = "simple.pickle"

optimism_vals = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# check if the file exists
import os
if os.path.exists(save_file):
    # load the data
    with open(save_file, 'rb') as f:
        experiments = pickle.load(f)

else:
    experiments = []

    for optimism in optimism_vals:
        print(end=f"optimism {optimism:5} steps = ")
        samples = []
        for _ in range(trials):
            dqn = DQN(state_dim=n_state, action_dim=n_action)
            samples.append( q_learning(env=env, model=dqn, episodes=episodes, optimism=optimism) )
            print(end=f"{len(samples[-1]):3} ")
        print()
        experiments.append(samples)
        # save the data
        with open(save_file, 'wb') as f:
            pickle.dump(experiments, f)

averages = []
for experiment in experiments:
    average = np.array([ x + [x[-1]]*(episodes-len(x)) for x in experiment ]).mean(0).tolist()
    averages.append(average)

plot_many(averages, titles=[ f"optimism {x}" for x in optimism_vals])

