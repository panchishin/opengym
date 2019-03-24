import pickle
import gym
import numpy as np
from geneticalgorithm import GeneticAlgorithm
from neural_pheno import Phenotype
from math import floor

phenotype_population = 50
file_name = 'data/mountaincar.pickle'

envir = gym.make('MountainCar-v0')

def oneTrial(phenotype, init_actions=[], demo=False) :
    state = envir.reset()
    previous_state = state
    total_reward = -100.0

    for count in range(501) :
        if demo :
            envir.render()

        if count < len(init_actions) :
            action = init_actions[count]
        else :
            action = np.argmax( phenotype.compute([state[0],state[1],state[0]-previous_state[0],state[1]-previous_state[1]] ) )

        state, reward, done, info = envir.step(action)
        previous_state = state
        total_reward = max(total_reward,state[1])

        if done :
            return total_reward

    return total_reward

# result = oneTrial(Phenotype(shape=[2,5,5,3]) , [] , demo=True)


def fitnessFunction(phenotype) :
    return ( oneTrial(phenotype) + oneTrial(phenotype,[0,0,0,0,0]) + oneTrial(phenotype,[1,1,1,1,1]) ) / 3.0


population = [ Phenotype(shape=[4,5,5,3]) for i in range(phenotype_population) ]

ga = GeneticAlgorithm(
    mutationFunction = lambda phenotype: phenotype.mutate(),
    crossoverFunction = lambda phenotype,other: phenotype.crossover(other),
    fitnessFunction = fitnessFunction,
    population = population,
    populationSize = phenotype_population
    )



print("")
print("-------------------------------------")
print("")
print("Initialize",phenotype_population,"agents for evolution.")
print("Before evolving the best score is {:>8f}, and mean score is {:>8f}".format(ga.bestScore(),ga.meanScore()))
print("")
print("Starting the evolution process")
print("")
try :
    for generation in range(0,101) :
        if generation % 5 == 0:
            print("generation {:>3}, best score is {:>8f}, and mean score is {:>8f}".format(generation,ga.bestScore(),ga.meanScore()))
            oneTrial(phenotype=ga.bestPhenotype(),demo=True)
            if ga.bestScore() >= 499 :
                print("Found a successful candidate")
                for count in range(3,0,-1) :
                    print("Demoing it {} more times".format(count))
                    oneTrial(phenotype=ga.bestPhenotype(),demo=True)
                break
        ga.evolve()
except :
    pass


envir.close()

