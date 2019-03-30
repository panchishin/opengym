import gym
import math
import numpy as np
from random import random, choice
import os
import geneticalgorithm as ga
from PIL import Image, ImageDraw
from array2gif import write_gif

file_name = "data/cartpole_nn"

env = gym.make('CartPole-v1')
action_list = [0,1]


def createLabel(text, width=80, height=40) :
    img = Image.new('RGB', (width, height), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((12,12), " " + text, fill=(255,255,255))
    return np.asarray(img)


def oneTrial(phenotype, init_actions=[], demo=False, label="Unlabeled") :
    state = env.reset()
    total_reward = 0
    video = []

    for count in range(501) :
        if demo :
            video.append( env.render(mode='rgb_array') )


        if count < len(init_actions) :
            action = init_actions[count]
        else :
            action = int(round( phenotype.compute(state)[0] ))

        state, reward, done, info = env.step(action)
        total_reward += reward

        if done : 
            break

    if demo :
        for i in range(30-count) :
            video.append( video[-1] )
        video = np.array(video)[:,300:700:4,::4]  # reshape from 800x1200 to 100x300
        video[:,0:40,0:160,:] -= createLabel(text=label + " score " + str(int(total_reward)),width=160,height=40)
        if total_reward < 200 :
            video[-15:,0:40,0:160,1:] = 64
        elif total_reward >= 400 :
            video[60:,0:40,0:160,0] = 64
            video[60:,0:40,0:160,2] = 64
        return video

    else :
        return total_reward


def fitnessFunction(args) :
    phenotype = ga.Phenotype(shape=[4,5,1],data=args)
    return -1 * ( oneTrial(phenotype) ) # + oneTrial(phenotype,[0,0,0,0,0]) + oneTrial(phenotype,[1,1,1,1,1]) ) / 3.0


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

videos = []
for i in range(0,20) :
    best = fmin(fitnessFunction, space, algo=tpe.suggest, trials=trials, max_evals=max(1,i*10), show_progressbar=False)
    args = space_eval(space, best)
    score = int(-1*fitnessFunction(args))
    print(i*10,"-->",score) 
    phenotype = ga.Phenotype(shape=[4,5,1],data=args)
    videos.append( oneTrial(phenotype,demo=True,label="Trial " + str(max(1,i*10))) )
    if score >= 500 :
        break

env.close()

print("Creating video")
video = np.concatenate( videos , 0 )
write_gif(video[::2],"pics/cartpole_hp.gif",fps=15)
print("Done")



env.close()
#env.close()