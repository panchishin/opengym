import math
import pickle
import torch
from torch.autograd import Variable
from plottool import plot_res, plot_many
import random
import numpy as np


class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, *, state_dim, action_dim, hidden_dim=32, lr=0.005):
            self.loss = torch.nn.MSELoss()
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


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def q_learning(*, env, model, episodes, 
               gamma=0.95, epsilon=0.1, eps_decay=0.99,
               error_threshold=0.03, verbose=False):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    win_count = 0
    error_threshold = math.pow(error_threshold,2)

    for episode in range(1,episodes+1):
        # Reset state
        state = env.reset()[0]
        done = False
        total = 0
        model_updates = 0
        
        while not done and total < 1500:
            # Epsilon-greedy
            action = env.action_space.sample() if random.random() < epsilon else torch.argmax(model.predict(state)).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)[:3]
            if done: reward = 0
            if total >= 1500: done = True
            
            # Update total and memory
            total += reward
            q_values = model.predict(state).tolist()
            q_values_next = model.predict(next_state)
            predicted_reward = torch.max(q_values_next).item()

            if math.pow(predicted_reward - reward,2) > error_threshold:
                q_values[action] = reward if done else (1-gamma)*reward + gamma*predicted_reward
                model.update(state, q_values)
                model_updates += 1
        
            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        
        if verbose:
            print(f"episode: {episode}, epsilon {int(epsilon*100)}/100 model updates {model_updates} reward: {int(total)} " + "*"*int(total/20))

        if total >= 1500:
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
        samples = pickle.load(f)

else:
    samples = []
    for _ in range(trials):
        dqn = DQN(state_dim=n_state, action_dim=n_action)
        samples.append( q_learning(env=env, model=dqn, episodes=episodes) )
        print("trial", _,"steps", len(samples[-1]))

    # save the data
    with open(save_file, 'wb') as f:
        pickle.dump(samples, f)

average = np.array([ x + [x[-1]]*(episodes-len(x)) for x in samples ]).mean(0).tolist()
samples.append(average)
plot_many(samples, titles=[f"trial{x}" for x in range(trials)] + ["average"])

