import gym
envir = gym.make('CartPole-v1')

state = envir.reset()
print("envir reset state :",state)
print("state space high :",envir.observation_space.high)
print("state space low  :",envir.observation_space.low)

action_space = envir.action_space
print("envir action space :",action_space)

state, reward, done, info = envir.step(envir.action_space.sample())
death = -1000


class Step :

    def __init__( self, state, action, reward ):
        self.state = state
        self.action = action
        self.reward = reward

    def __repr__(self):
        return "{} {} {}".format(str(self.state)  , self.action , self.reward)



def dosome(some=200):
    state = envir.reset()
    data = []
    for count in range(some):
        action = envir.action_space.sample()
        # envir.render()
        next_state, reward, done, info = envir.step(action)
        data.append( Step(state=state,action=action,reward=reward) )
        state = next_state
        if done : 
            data.append( Step(state=state,action=envir.action_space.sample(),reward=death) )
            break
    return data


def run() :
    items = dosome()
    for a in items:
        print(a)
    print("There are",len(items),"items")

run()