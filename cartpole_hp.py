import gym
import math
import numpy as np
from random import random, choice
import os
import geneticalgorithm as ga

file_name = "data/cartpole_nn"

env = gym.make('CartPole-v1')
action_list = [0,1]


def oneTrial(phenotype, init_actions=[], demo=False) :
    state = env.reset()
    total_reward = 0

    for count in range(501) :
        if demo :
            env.render()

        if count < len(init_actions) :
            action = init_actions[count]
        else :
            action = int(round( phenotype.compute(state)[0] ))

        state, reward, done, info = env.step(action)
        total_reward += reward

        if done :
            return total_reward

    return total_reward


def fitnessFunction(args) :
    phenotype = ga.Phenotype(shape=[4,5,1],data=args)
    return -1 * ( oneTrial(phenotype) + oneTrial(phenotype,[0,0,0,0,0]) + oneTrial(phenotype,[1,1,1,1,1]) ) / 3.0


from hyperopt import hp
space = [
    [ [ hp.uniform('l0m'+str(i),-5,5) for i in range(4*5) ] , [ hp.uniform('l1m'+str(i),-5,5) for i in range(5) ] ],
    [ [ hp.uniform('l0b'+str(i),-.5,.5) for i in range(5) ] , [ hp.uniform('l1b0',-.5,.5) ] ]
]




print("")
print("Starting the training process")
print("")

from hyperopt import fmin, tpe, space_eval, Trials
trials = Trials()

print("With an iterative call")
for i in range(1,10) :
    best = fmin(fitnessFunction, space, algo=tpe.suggest, trials=trials, max_evals=i*10, show_progressbar=False)
    args = space_eval(space, best)
    print(i*10,"-->",int(-1*fitnessFunction(args))) 
    phenotype = ga.Phenotype(shape=[4,5,1],data=args)
    oneTrial(phenotype,demo=True)



env.close()
#env.close()