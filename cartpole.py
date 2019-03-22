import pickle
import gym
from geneticalgorithm import GeneticAlgorithm
from neural_pheno import Phenotype

phenotype_population = 50
file_name = 'data/cartpole_best.pickle'

envir = gym.make('CartPole-v1')

def oneTrial(phenotype, init_actions=[], demo=False) :
    state = envir.reset()
    total_reward = 0

    for count in range(501) :
        if demo :
            envir.render()

        if count < len(init_actions) :
            action = init_actions[count]
        else :
            action = int(round( phenotype.compute(state) ))

        state, reward, done, info = envir.step(action)
        total_reward += reward

        if done :
            return total_reward

    return total_reward


def fitnessFunction(phenotype) :
    return ( oneTrial(phenotype) + oneTrial(phenotype,[0,0,0,0,0]) + oneTrial(phenotype,[1,1,1,1,1]) ) / 3.0


population = [ Phenotype() for i in range(phenotype_population) ]
try :
    with open(files_name, 'rb') as handle:
        population = [pickle.load(handle)]
except :
    pass

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
print("Before evolving the best score is {:>3}, and mean score is {:>3}".format(int(ga.bestScore()),int(ga.meanScore())))
print("")
print("Starting the evolution process")
print("")
try :
    for generation in range(0,101) :
        if generation % 5 == 0:
            print("generation {:>3}, best score is {:>3}, and mean score is {:>3}".format(generation,int(ga.bestScore()),int(ga.meanScore())))
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

with open(files_name, 'wb') as handle:
    pickle.dump(ga.bestPhenotype(),handle)

envir.close()

