import functools
from random import random
from random import choice
import math

message = """
This is the cliff problem.\nThe virtual world consists of a 7x3 world like so:
 +-------+\n |ScccccR|\n |       |\n |       |\n +-------+
The bot starts at 'S' and is trying to get to 'R'.  If the bot touches a 'c' it falls off the cliff.
Falling off the cliff is -10,000 points, the reward is 1,000 points, and everything else is -1 point.
After the bot falls or gets a reward it starts again.  To make matters more challenging, 20% of the
time the bot will move randomly, increasing the odds that it falls off the cliff.
It takes 12 moves to take the 'long' way around the cliff, so a score of 80 is near perfect
and assumes that no random movements delay the bot.
"""

print(message)

map = [ 
  [     -1., -1., -1. ],
  [ -10000., -1., -1. ],
  [ -10000., -1., -1. ],
  [ -10000., -1., -1. ],
  [ -10000., -1., -1. ],
  [ -10000., -1., -1. ],
  [   1000., -1., -1. ] 
]

width = len(map)
height = len(map[0])

# The possible actions
action_list = ['up','down','right','left','hold']
    
def move(location,action) :
    if action == 'up'    : return { 'h' : min(location['h'] + 1, height-1), 'w' : location['w'] }
    if action == 'down'  : return { 'h' : max(location['h'] - 1, 0),        'w' : location['w'] }
    if action == 'left'  : return { 'h' : location['h'] , 'w' : min(location['w'] + 1, width-1) }
    if action == 'right' : return { 'h' : location['h'] , 'w' : max(location['w'] - 1, 0)       }
    return { 'h':location['h'], 'w':location['w'] }


import sarsa

sarsa = sarsa.Sarsa()

# keep track of where the virtual bot is
location = None
action = None
reward = 0.0

# let it learn by trying a lot.  This is the number of moves, not the number of games.
trials_max = 8191

# we keep track of the last several rewards to calculate average reward
reward_history = []
def averageReward(reward_history) :
    return 1.0 * functools.reduce( lambda x,y : 0.0+x+y , reward_history ) / len(reward_history)


last_full_run = [] # movement of last full game
current_run = [] # current game

for trials in range(trials_max) :

    # if the bot touches something other than regular 'ground' then restart.
    if reward != -1.0 :
        location = {'w':0,'h':0}
        action = 'hold'
        last_full_run = current_run
        current_run = [location]


    next_location = move( location, action )
    next_action = sarsa.chooseAction( next_location, action_list )

    # 20% of the time the bot does not go where it wants but instead does something random
    if ( random() <= 0.2 ) and ( trials < trials_max - 100 ) :
        next_action = choice(action_list)
        #pass

    # get reward from map, see top
    reward = map[next_location['w']][next_location['h']];

    sarsa.update(location,action,reward,next_location,next_action)

    # set the current location and action for the next step
    location = next_location;
    action = next_action;

    current_run.append(location)

    # add the reward to the history so we can calculate an average
    reward_history.append(reward)
    if len(reward_history) > 800 : # only keep a bit of recent history
        reward_history = reward_history[1:]
    

    if math.log2(trials+1) % 1 == 0 and trials >= 64 :
        print("Move",trials,", average reward per move",averageReward(reward_history),"for the last",len(reward_history),"moves")
  

average_reward = averageReward(reward_history)

if average_reward >= 70 :
    print("After",trials_max,"moves the SARSA RL algorithm found a solution to the\n",
    "cliff problem and accumulated an average of",average_reward,"points per move.\n",
    "These results are good and expected.")

elif average_reward >= 50 :
    print("These results are fair.  Try running the simulation again.")

else :
    print("These results are very poor.  Try running the simulation again.")


print("Here is the last run.")

for i in last_full_run :
    print(i)

map = [ 
  [ 'S', ' ', ' ' ],
  [ 'c', ' ', ' ' ],
  [ 'c', ' ', ' ' ],
  [ 'c', ' ', ' ' ],
  [ 'c', ' ', ' ' ],
  [ 'c', ' ', ' ' ],
  [ 'R', ' ', ' ' ] 
]


def printMap(map) :
    print("+-------+")
    for height in range(len(map[0])) : 
        row = []
        for width in range(len(map)) :
            row.append( map[width][height] )
        print("|" + "".join(row) + "|")
    print("+-------+")


print("An empty map")
printMap(map)

for i in last_full_run[:50] : 
    map[ i['w'] ][ i['h'] ] = "x"

print("The last path taken marked by 'x'")
printMap(map)
