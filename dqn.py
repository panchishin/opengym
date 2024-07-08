import math
import torch
from torch.autograd import Variable
import random
import numpy as np

def clamp(new_value,clamp,old_value):
    return max(old_value-clamp, min(new_value, old_value+clamp))

class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, *, state_dim:int, action_dim:int, hidden_dim:int=16, lr:float=0.005):
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

    def optimistic_init(self, env, optimism):
        """ Train the network to be optimistic about the rewards for random states"""
        optimistic = np.array([optimism]*self.action_dim)
        for _ in range(1000):
            state = env.reset()[0]
            state = np.array(state)
            state += np.random.normal(0, 0.5, state.shape)
            state += np.random.normal(0, 0.5, state.shape)
            state += np.random.normal(0, 0.5, state.shape)
            state += np.random.normal(0, 0.5, state.shape)
            self.update(state, optimistic)

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def clone(self):
        return DQN(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, lr=self.lr)

def q_learning(*, env, model:DQN, win_score:float=1000, episodes:int=500,
               gamma:float=0.95, epsilon:float=0.5, eps_decay:float=0.9, clip_size:float=0.2, error_threshold:float=0.03,
               verbose:bool=False, optimistic_init:bool=True, use_regret:bool=False, optimism:float=0.9):
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
        model.optimistic_init(env, optimism=optimism)
        if use_regret:
            regret_model.optimistic_init(env, optimism=optimism)

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

