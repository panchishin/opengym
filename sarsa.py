# SARSA

import json
from random import random
from random import choice
import math
import functools


def cloneJSON( item ):
    return json.loads(json.dumps(item))


def setReward( weights, state, action, reward ):
    state = json.dumps(state)
    weights[state] = {} if state not in weights else weights[state]
    weights[state][action] = reward


def getRewards( weights, state, action_list, defaultReward ):
    actions = {} if json.dumps(state) not in weights else weights[json.dumps(state)]
    result = {}
    for action in action_list :
        if action in actions :
            result[action] = actions[action]
        else :
            result[action] = defaultReward    
    return result


def sarsaEquation(state0, action0, reward1, state1, action1, alpha, gamma, weights, defaultReward, getRewards, setReward) :
    # formula : ( 1 - a )Q(t) + (a)( r + yQ(t+1) )

    a = alpha
    Qt0 = getRewards( weights, state0, [action0], defaultReward )[action0]
    Qt1 = getRewards( weights, state1, [action1], defaultReward )[action1]
    r = reward1
    y = gamma

    result = (1-a)*Qt0 + a*(r+y*Qt1)
    setReward( weights, state0, action0, result )
    return result



def randomPolicy( actions, epsilon ) :
    actions = [a for a in actions.keys()]
    return choice(actions)


def greedyPolicy( actions, epsilon ) :
    best_score = functools.reduce( lambda x,y : max(x,y) , actions.values() )
    return next(filter( lambda key : actions[key] == best_score , actions.keys() ))


def epsilonGreedyPolicy( actions, epsilon ) :
    if ( random() <= epsilon ) :
        return randomPolicy(actions,epsilon)
    else :
        return greedyPolicy(actions,epsilon)
    

policies = {
    'greedy' : greedyPolicy,
    'epsilonGreedy' : epsilonGreedyPolicy,
    'random' : randomPolicy
}


defaults = {
    'alpha' : 0.2,     # default to a low(-ish) learning rate
    'gamma' : 0.8,     # default of a high(-ish) dependance on future expectation
    'defaultReward' : 0,
    'epsilon' : 0.001,
    'policy' : 'greedy'
}

class Sarsa:
    def __init__(self, config=None):
        global defaults
        self.config = config or cloneJSON(defaults)
        self.weights = {}

    def getRewards(self, state, action_list) :
        return cloneJSON(getRewards(self.weights,state,action_list,self.config['defaultReward']))

    def update (self, state0, action0, reward1, state1, action1) :
        return sarsaEquation(state0,action0,reward1,state1,action1,
            self.config['alpha'],self.config['gamma'],
            self.weights,self.config['defaultReward'],getRewards,setReward)
    
    def chooseAction(self, state, action_list) :
        global policies
        actions = getRewards(self.weights,state,action_list,self.config['defaultReward'])
        return policies[self.config['policy']](actions,self.config['epsilon'])
    


class TransformState:
    def __init__(self, transFunc, config=None, impl=None):
        if impl != None :
            self.impl = impl
        else :
            self.impl = Sarsa(config)
        self.transFunc = transFunc

    def getRewards(self, state, action_list) :
        return self.impl.getRewards(self.transFunc(state),action_list)

    def update(self, state0, action0, reward1, state1, action1) :
        return self.impl.update(self.transFunc(state0), action0, reward1, self.transFunc(state1), action1)

    def chooseAction(self, state, action_list):
        return self.impl.chooseAction(self.transFunc(state), action_list)


class Combine:
    def __init__(self, implList, config=None):
        self.implList = implList
        self.config = config or cloneJSON(defaults)

    def update(self, state0, action0, reward1, state1, action1) :
        for impl in self.implList :
            return impl.update(state0, action0, reward1, state1, action1)

    def chooseAction(self, state, action_list):
        rewards = [ impl.getRewards(state,action_list) for impl in self.implList ]
        actions = {}
        for action in action_list :
            actions[action] = 0
        for reward in rewards :
            for action in actionList :
                if action in reward :
                    actions[action] += reward[action]
        return policies[self.config['policy']](actions,self.config['epsilon'])


