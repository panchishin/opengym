import pickle
import gym
from geneticalgorithm import GeneticAlgorithm
from neural_pheno import Phenotype

envir = gym.make('CartPole-v1')
action_list = [0,1]
file_name = 'data/cartpole_sarsa_best.pickle'

def digitizeState(state) :
    return tuple([ int(round(x*5.)) for x in state ])

def oneTrial(agent, init_actions=[], demo=False) :
    state0 = state1 = digitizeState(envir.reset())
    total_reward = 0
    reward0 = None

    for count in range(501) :
        if demo :
            envir.render()

        if count < len(init_actions) :
            action1 = init_actions[count]
        else :
            action1 = agent.chooseAction( state0, action_list )

        state2, reward1, done, info = envir.step(action1)
        state2 = digitizeState(state2)

        if reward0 != None :
            #if demo :
            #    print(state0,action0,reward0,"(",agent.getRewards( state0, action_list ),")",state1,action1)
            reward_temp = -1000 if done else reward0
            agent.update(state0,action0,reward_temp,state1,action1)
        state0 = state1
        state1 = state2
        action0 = action1
        reward0 = reward1

        total_reward += reward0

        if done :
            return total_reward

    return total_reward




print("")
print("-------------------------------------")
print("")
print("Initialize SARSA agent.")

import sarsa
agent_settings = {
    'alpha' : 0.5,     # default to a low(-ish) learning rate
    'gamma' : 0.8,     # default of a high(-ish) dependance on future expectation
    'defaultReward' : 0,
    'epsilon' : 0.001,
    'policy' : 'epsilonGreedy'
}
agent = sarsa.Sarsa(config = agent_settings)

try :
    with open(file_name, 'rb') as handle:
        agent = pickle.load(handle)
except :
    print("FAILED TO OPEN SAVE FILE")


print("")
print("Starting the training process")
print("")
lambda_score = 0
try :
    for session in range(0,90001) :
            score = oneTrial(agent=agent)
            lambda_score = 1. * ( min(9.,session+1) * lambda_score + score ) / min(10.,session+1)
            if session % 1000 == 0:
                print("session {:>5}, score is {:>5}".format(session,int(lambda_score)))
                oneTrial(agent=agent,demo=True)
                if score >= 499 :
                    break
except :
    pass

oneTrial(agent=agent,demo=True)


with open(file_name, 'wb') as handle:
    pickle.dump(agent,handle)


print("Finished sessions.")
for count in range(3,0,-1) :
    print("Demoing agent {} more times".format(count))
    oneTrial(agent=agent,demo=True)

envir.close()