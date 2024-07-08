"""
Experiements to run
 - use a bottlneck layer for identifying how many times the network has seen this state to calculate the regret for exploration (instead of epsilon greedy)
 - use a NN that given the S,A predicts S'.  Use this to calculate the S' that is least like the current state and use that as the exploration
 - or like diffussion, predict delta S
 - use a second NN to predict the reward given the S,A and use the difference in predicted Q to the first NN to estimate regret for exploration
"""

import math
import pickle
import torch
from torch.autograd import Variable
from plottool import plot_res, plot_many, plot_mean_std
from plottool import scatter_plot
import random
import numpy as np

def clamp(new_value,clamp,old_value):
    return max(old_value-clamp, min(new_value,old_value+clamp))

class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, *, state_dim, action_dim, hidden_dim=32, lr=0.005):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.lr = lr

            self.loss = torch.nn.MSELoss() # just as good as huber loss
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),   torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim),  torch.nn.Tanh(),
                            torch.nn.Linear(hidden_dim, action_dim),  torch.nn.Sigmoid(),
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        self.optimizer.zero_grad()
        loss = self.loss(y_pred, Variable(torch.Tensor(y)))
        loss.backward()
        self.optimizer.step()

    def optimistic_init(self, env):
        """ Train the network to be optimistic about the rewards for random states"""
        optimistic = np.array([0.98]*self.action_dim)
        for _ in range(1000):
            state = env.reset()[0]
            state = np.array(state)
            state += np.random.normal(0, 0.5, state.shape)
            self.update(state, optimistic)

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def clone(self):
        return DQN(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, lr=self.lr)

def q_learning(*, env, model, win_score=1000, episodes=500,
               gamma=0.95, epsilon=0.5, eps_decay=0.9, clip_size=0.2, error_threshold=0.03,
               verbose=False, optimistic_init=True, use_regret=False):
    """Deep Q Learning algorithm using the DQN.
    gamma = the discount factor for future rewards
    epsilon = the exploration rate
    eps_decay = the rate at which epsilon decays
    clip_size = the maximum size of the update to the Q value
    error_threshold = the minimum error required to update the Q value
    """
    final = []
    win_count = 0
    error_threshold = math.pow(error_threshold,2)

    if use_regret:
        regret_model = model.clone()

    if optimistic_init:
        model.optimistic_init(env)
        if use_regret:
            regret_model.optimistic_init(env)

    for episode in range(1,episodes+1):
        # Reset state
        state = env.reset()[0]
        done = False
        total = 0
        model_updates = 0
        
        while not done and total < win_score:
            # Epsilon-greedy
            explore = random.random() < epsilon
            if explore:
                if use_regret:
                    regret_predict = regret_model.predict(state)
                    model_predict = model.predict(state)
                    action = torch.argmax( np.square(regret_predict - model_predict) ).item()
                else:
                    action = env.action_space.sample()
            else:
                action = torch.argmax(model.predict(state)).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)[:3]
            if done: reward = 0
            if total >= win_score: done = True
            
            # Update total and memory
            total += reward
            q_values = model.predict(state).tolist()
            q_values_next = model.predict(next_state)
            predicted_reward = torch.max(q_values_next).item()

            if math.pow(predicted_reward - reward,2) > error_threshold:
                if done:
                    q_values[action] = reward
                else:
                    updated_value = (1-gamma)*reward + gamma*predicted_reward
                    q_values[action] = clamp(new_value=updated_value,clamp=clip_size,old_value=q_values[action])

                model.update(state, q_values)
                model_updates += 1

            if explore and use_regret:
                action_value = q_values[action]
                q_values = regret_model.predict(state).tolist()
                q_values_next = regret_model.predict(next_state)
                q_values[action] = action_value
                regret_model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        
        if verbose:
            print(f"episode: {episode}, epsilon {int(epsilon*100)}/100 model updates {model_updates} reward: {int(total)} " + "*"*int(total/20))

        if total >= win_score:
            win_count += 1
            if win_count >= 10:
                if verbose:
                    print(f"episode: {episode}, Task solved")
                break
        else:
            win_count = 0

    return final



import gym
env = gym.make('CartPole-v1')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
episodes = 500
trials = 5

save_file = "simple.pickle"

# check if the file exists
import os
if os.path.exists(save_file):
    # load the data
    with open(save_file, 'rb') as f:
        experiments = pickle.load(f)

else:
    experiments = []

    samples = []
    for _ in range(trials):
        dqn = DQN(state_dim=n_state, action_dim=n_action)
        samples.append( q_learning(env=env, model=dqn, episodes=episodes, use_regret=False) )
        print("trial", _,"steps", len(samples[-1]))
    experiments.append(samples)
 
    samples = []
    for _ in range(trials):
        dqn = DQN(state_dim=n_state, action_dim=n_action)
        samples.append( q_learning(env=env, model=dqn, episodes=episodes, use_regret=True) )
        print("trial", _,"steps", len(samples[-1]))
    experiments.append(samples)

    # save the data
    with open(save_file, 'wb') as f:
        pickle.dump(experiments, f)

averages = []
for experiment in experiments:
    average = np.array([ x + [x[-1]]*(episodes-len(x)) for x in experiment ]).mean(0).tolist()
    averages.append(average)

plot_many(averages, titles=["Simple DQN", "DQN with Regret"])

