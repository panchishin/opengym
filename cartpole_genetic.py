import gym
from geneticalgorithm import GeneticAlgorithm, Phenotype

phenotype_population = 50
file_name = 'data/cartpole_genetic.pickle'

def oneTrial(phenotype, init_actions=[], demo=False) :
    if demo :
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1')
    state = env.reset()

    total_reward = 0

    for count in range(501) :
        if demo :
            env.render()

        if count < len(init_actions) :
            action = init_actions[count]
        else :
            action = int(round( phenotype.compute(state)[0] ))

        temp = env.step(action)
        state, reward, done, info = temp[:4]
        total_reward += reward

        if done :
            return total_reward

    env.close()
    return total_reward


def fitnessFunction(phenotype) :
    return ( oneTrial(phenotype) + oneTrial(phenotype,[0,0,0,0,0]) + oneTrial(phenotype,[1,1,1,1,1]) ) / 3.0


population = [ Phenotype() for i in range(phenotype_population) ]

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
best_score_list = []
mean_score_list = []
for generation in range(0,501) :
        if generation % 5 == 0:
            best_score_list.append(ga.bestScore())
            mean_score_list.append(ga.meanScore())
            print("generation {:>3}, best score is {:>3}, and mean score is {:>3}".format(generation,int(best_score_list[-1]),int(mean_score_list[-1])))
            # oneTrial(phenotype=ga.bestPhenotype(),demo=True)
            if ga.bestScore() >= 299 :
                print("Found a successful candidate")
                for count in range(3,0,-1) :
                    print("Demoing it {} more times".format(count))
                    oneTrial(phenotype=ga.bestPhenotype(),demo=True)
                break
        ga.evolve()

from plottool import plot_res
plot_res(best_score_list, "Best score")
plot_res(mean_score_list, "Mean score")
