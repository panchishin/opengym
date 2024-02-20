import math
import torch
from torch.autograd import Variable
from plottool import plot_res
import random
import time
import copy

class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim, lr=0.01):
            self.loss = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            # torch.nn.Tanh(),
                            torch.nn.Linear(hidden_dim, hidden_dim),
                            # torch.nn.LeakyReLU(),
                            torch.nn.Tanh(),
                            torch.nn.Linear(hidden_dim, action_dim),
                            torch.nn.Sigmoid(),
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
        if len(memory)>=size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            #Update q values
            all_q_values[range(len(all_q_values)),actions]=(1-gamma)*rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            
            self.update(states.tolist(), all_q_values.tolist())


class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if True:  #len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, min(size, len(memory)))
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = (1-gamma)*reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets)



def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=10,
               title = 'DQL', double=False, 
               n_update=10, verbose=True, memory=None):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = memory or []

    for episode in range(1,episodes+1):
        if double:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        
        # Reset state
        state = env.reset()[0]
        done = False
        total = 0
        model_updates = 0

        total_error = 0
        
        while not done and total < 500:
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
            
            # Update total and memory
            total += reward
            q_values = model.predict(state).tolist()
            q_values_next = model.predict(next_state)
            predicted_reward = torch.max(q_values_next).item()

            total_error += math.pow(predicted_reward - reward,2)
            # if math.pow(predicted_reward - reward,2) > 0.04:
            if total_error > 0.04:
                total_error = 0
                if replay:
                    memory.append((state, action, next_state, reward, done))
                    memory = memory[-replay_size*2:]
                    model.replay(memory, replay_size, gamma)
                else:
                    q_values[action] = reward if done else (1-gamma)*reward + gamma*predicted_reward
                    model.update(state, q_values)
                model_updates += 1
            else :
                total_error *= 0.9

            if done and not replay:
                break

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        
        if verbose:
            print(f"episode: {episode}, epsilon {int(epsilon*1000)}/1000 model updates {model_updates} reward: {int(total)} " + "*"*int(total/5))


    plot_res(final, title)

    return memory



if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')

    # Number of states
    n_state = env.observation_space.shape[0]
    # Number of actions
    n_action = env.action_space.n
    # Number of episodes
    episodes = 500
    # Number of hidden nodes in the DQN
    n_hidden = 64
    # Learning rate
    lr = 0.001
    epsilon = 0.1
    gamma = 0.9


    simple_dqn = DQN(n_state, n_action, n_hidden, lr)
    simple = q_learning(env, simple_dqn, episodes, gamma=gamma, epsilon=epsilon, title='Simple DQL')

    dqn_replay = DQN_replay(n_state, n_action, n_hidden, lr)
    replay = q_learning(env, dqn_replay, episodes, gamma=gamma, epsilon=epsilon, replay=True, title='DQL with Replay')

    dqn_double = DQN_double(n_state, n_action, n_hidden, lr)
    memory =  q_learning(env, dqn_double, episodes, gamma=gamma, epsilon=epsilon, replay=True, double=True,title='Double DQL with Replay', n_update=10)

