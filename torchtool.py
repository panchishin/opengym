import math
import pickle
import torch
from torch.autograd import Variable
from plottool import plot_res, plot_many
import random
import numpy as np


class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=0.005):
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
        loss = self.loss(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))


# Expand DQL class with a replay function.
class DQN_replay(DQN):
    
    def replay(self, memory, size, gamma=0.9):
        """New replay function"""
        #Try to improve replay speed
        if True:  #len(memory) >= size:
            # Sample experiences from the agent's memory
            batch = random.sample(memory, min(size, len(memory)))
            states , actions , next_states , rewards , is_dones = list(map(list, zip(*batch))) #Transpose batch list
            
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            # Update q values
            all_q_values[range(len(all_q_values)),actions]=(1-gamma)*rewards+gamma*torch.max(all_q_values_next, axis=1).values
            # add rewards to the q values of the last states
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
            
            self.update(states.tolist(), all_q_values.tolist())


class DQN_double():
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.dqn_replay = DQN_replay(state_dim, action_dim, hidden_dim, lr)
        self.dqn = DQN(state_dim, action_dim, hidden_dim, lr)

    def target_update(self):
        ''' Update target network with the model weights.'''
        self.dqn.model.load_state_dict(self.dqn_replay.model.state_dict())

    def predict(self, state):
        return self.dqn.predict(state)

    def replay(self, memory, size, gamma=0.9):
        return self.dqn_replay.replay(memory, size, gamma)



def q_learning(env, model, episodes, gamma=0.95, 
               epsilon=0.1, eps_decay=0.99,
               replay=False, replay_size=20,
               double=False, error_threshold=0.03,
               n_update=10, verbose=False, memory=None):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = memory or []
    win_count = 0
    error_threshold = math.pow(error_threshold,2)

    for episode in range(1,episodes+1):
        # Reset state
        state = env.reset()[0]
        done = False
        total = 0
        model_updates = 0
        
        while not done and total < 1500:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)[:3]
            if done:
                reward = 0
            if total >= 1500:
                done = True
            
            # Update total and memory
            total += reward
            q_values = model.predict(state).tolist()
            q_values_next = model.predict(next_state)
            predicted_reward = torch.max(q_values_next).item()

            if math.pow(predicted_reward - reward,2) > error_threshold:
                if replay:
                    memory.append((state, action, next_state, reward, done))
                    memory = memory[-replay_size*2:]
                    model.replay(memory, replay_size, gamma)
                else:
                    q_values[action] = reward if done else (1-gamma)*reward + gamma*predicted_reward
                    model.update(state, q_values)
                model_updates += 1

            if double:
                # Update target network every n_update steps
                if (episode+model_updates) % n_update == 0:
                    model.target_update()
        

            if done and not replay:
                break

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        
        if verbose:
            print(f"episode: {episode}, epsilon {int(epsilon*100)}/100 model updates {model_updates} reward: {int(total)} " + "*"*int(total/20))

        if total >= 1500:
            win_count += 1
            if win_count >= 10:
                print(f"Task solved in {episode} episodes!")
                break
        else:
            win_count = 0

    return final



import gym
env = gym.make('CartPole-v1')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
episodes = 500
n_hidden = 32
lr = 0.005
epsilon = 0.1
gamma = 0.95
trials = 20
error_threshold = 0.03

save_file = "simple.pickle"

# check if the file exists
import os
if os.path.exists(save_file):
    # load the data
    with open(save_file, 'rb') as f:
        samples = pickle.load(f)

else:
    samples = []
    simple_data = []
    for _ in range(trials):
        dqn = DQN(n_state, n_action, n_hidden, lr)
        simple_data.append( q_learning(env, dqn, episodes, gamma=gamma, epsilon=epsilon, error_threshold=error_threshold) )
        print("error_threshold", error_threshold, "trial", _,"steps", len(simple_data[-1]))
    samples.append(simple_data)

    # save the data
    with open(save_file, 'wb') as f:
        pickle.dump(samples, f)

data = []
for d in samples:
    data.append( np.array([ x + [x[-1]]*(episodes-len(x)) for x in d ]).mean(0).tolist() )

plot_many(samples[0], titles=[f"trial{x}" for x in range(trials)])

