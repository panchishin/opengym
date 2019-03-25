import pickle
import gym
import math
import sarsa

env = gym.make('CartPole-v1')
action_list = [0,1]
file_name = 'data/cartpole_sarsa_best.pickle'

def digitizeState(state) :
    return tuple([ int(round(x*5.)) for x in state ])

def oneTrial(agent, init_actions=[], demo=False) :
    state0 = state1 = digitizeState(env.reset())
    total_reward = 0
    reward0 = None
    replay = []

    for count in range(501) :
        if demo :
            env.render()

        if count < len(init_actions) :
            action1 = init_actions[count]
        else :
            action1 = agent.chooseAction( state1, action_list )

        state2, reward1, done, info = env.step(action1)
        state2 = digitizeState(state2)
        total_reward += reward1

        if reward0 != None :
            reward_temp = -1e8 if done and total_reward < 490 else reward0
            agent.update(state0,action0,reward_temp,state1,action1)
            replay.append([reward_temp,state1,action1])

        state0 = state1
        state1 = state2
        action0 = action1
        reward0 = reward1

        if done :
            for i in range(len(replay)-1,0,-1) :
                agent.update(replay[i-1][1],replay[i-1][2],
                    replay[i][0],replay[i][1],replay[i][2])
            return total_reward

    return total_reward




print("")
print("-------------------------------------")
print("")
print("Initialize SARSA agent.")


agent_settings = {
    'alpha' : 0.2,     # default to a low(-ish) learning rate
    'gamma' : 0.8,     # default of a high(-ish) dependance on future expectation
    'defaultReward' : 0,
    'epsilon' : 0.001,
    'policy' : 'epsilonGreedy'
}
basic_agent = sarsa.Sarsa(config = agent_settings)

agents = [basic_agent]
agents.append( sarsa.TransformState(lambda state:state[:2],sarsa.Sarsa(config = agent_settings)) )
agents.append( sarsa.TransformState(lambda state:state[2:],sarsa.Sarsa(config = agent_settings)) )
agents.append( sarsa.TransformState(lambda state:[state[0],state[2]],sarsa.Sarsa(config = agent_settings)) )
agents.append( sarsa.TransformState(lambda state:[state[1],state[3]],sarsa.Sarsa(config = agent_settings)) )
for i in range(4):
    agents.append( sarsa.TransformState(lambda state:state[i],sarsa.Sarsa(config = agent_settings)) )

combine = sarsa.Combine( agents , config=agent_settings )
agent = sarsa.TransformState(lambda state:tuple([ int(round(x*9.)) for x in state ]))

try :
    with open(file_name, 'rb') as handle:
        agent = pickle.load(handle)
except :
    print("FAILED TO OPEN SAVE FILE")


print("")
print("Starting the training process")
print("")
def doSession(sessions = 8192*2) :
    lambda_score = 0
    for session in range(1,sessions+1) :
        score = oneTrial(agent=agent)
        lambda_score = 1. * ( min(4.,session) * lambda_score + score ) / min(5.,session)
        if ( math.log2(session) % 1 == 0 or session % 1024 == 0 ) and session >= 16:
            print("session {:>5}, score is {:>5}".format(session,int(lambda_score)))
            oneTrial(agent=agent,demo=True)
            if score >= 499 :
                return


try :
    doSession()
except :
    print("Breaking out of training")
    pass

print("Writing agent to file.")
with open(file_name, 'wb') as handle:
    pickle.dump(agent,handle)


print("Finished sessions.")
for count in range(2,0,-1) :
    print("Demoing agent {} more times".format(count))
    oneTrial(agent=agent,demo=True)

env.close()