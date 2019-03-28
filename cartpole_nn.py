import gym
import math
import numpy as np
from random import random, choice
import os

file_name = "data/cartpole_nn"

env = gym.make('CartPole-v1')
action_list = [0,1]



print("")
print("-------------------------------------")
print("")
print("Initialize Tensorflow agent.")

alpha = 0.2   # default to a low(-ish) learning rate
gamma = 0.8   # default of a high(-ish) dependance on future expectation
epsilon = 0.1 # the chance of making a random action

import tensorflow as tf

np.set_printoptions(precision=2, suppress=True)
state_in = tf.placeholder(dtype=tf.float32, shape=[None,4], name="State")
action_in = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Action")
reward_in = tf.placeholder(dtype=tf.float32, shape=[None,1], name="Reward_ground")

concat_in = tf.concat([state_in,action_in], axis=1, name="Concat")
hidden_1 = tf.layers.dense(inputs=concat_in, units=5, activation='tanh', use_bias=True, name="H1")
hidden_2 = tf.layers.dense(inputs=hidden_1, units=5, activation='tanh', use_bias=True, name="H2")
reward_out = tf.layers.dense(inputs=hidden_2, units=1, activation=None, use_bias=True, name="Reward")

loss = tf.square(reward_in - reward_out)
train = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

sess = tf.Session()

if os.path.exists(file_name + ".index"):
    print("----------------- RESTORING FROM FILE -----------------")
    saver = tf.train.Saver()
    saver.restore(sess, file_name)
else :
    print("----------------- INITIALIZING VARS -----------------")
    sess.run(tf.global_variables_initializer())



def scoreAction( state, action ):
    return sess.run(reward_out , feed_dict={state_in:state.reshape(1,4),action_in:[[action]]})[0][0]

def updatePolicy( state, action, reward ):
    return sess.run(train , feed_dict={state_in:state.reshape(1,4),action_in:[[action]],reward_in:[[reward]]})


def oneTrial(demo=False) :
    state0 = state1 = env.reset()
    frames = 0

    for count in range(501) :
        if demo : env.render()

        action_scores = np.array([ scoreAction( state1, action_list[i]) for i in action_list ])
        action1 = action_scores.argmax()
        if not demo and random() <= epsilon :
            action1 = choice([0,1])

        state2, reward1, done, info = env.step(action1)
        frames += reward1

        if not demo and count > 0 :
            reward_temp = 0 if ( done and frames < 490 ) else reward0
            updatePolicy( state0,action0,reward_temp + action_scores[action1]*gamma )

        state0 = state1
        state1 = state2
        action0 = action1
        reward0 = reward1

        if done : return frames

    return frames





print("")
print("Starting the training process")
print("")
def doSession(sessions = 128*8) :
    lambda_score = 0
    for session in range(1,sessions+1) :
        score = oneTrial()
        lambda_score = 1. * ( (min(10.,session)-1.) * lambda_score + score ) / min(10.,session)
        if (score >= 499 and lambda_score >= 400 and session >= 16) or ( math.log2(session) % 1 == 0 or session % 64 == 0 ) and session >= 16:
            print("session {:>5}, score is {:>5}, λ 10 avg {:>5}".format(session,int(score),int(lambda_score)))
            oneTrial(demo=True)
            if score >= 499 and lambda_score >= 400 :
                print("WE DID IT!")
                return


doSession()
saver = tf.train.Saver()
saver.save(sess, file_name)

env.close()