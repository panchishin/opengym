import numpy as np
from random import choice, random
import gym
from geneticalgorithm import GeneticAlgorithm
import pickle

# ----------------------------------------------

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def genNewM(w,h):
    return 4. * ( np.random.rand(w,h) - 0.5 ) / w

def genNewB(h):
    return ( np.random.rand(h) - 0.5 ) / 5.0


class Phenotype :
    def __init__(self,clone=None) :

        if clone :
            self.ms = [ np.copy(clone.ms[i]) for i in range(len(clone.ms)) ]
            self.bs = [ np.copy(clone.bs[i]) for i in range(len(clone.bs)) ]
        else :
            self.ms = [ genNewM(4,5) , genNewM(5,5) , genNewM(5,1) ]
            self.bs = [ genNewB(5) ,   genNewB(5) ,   genNewB(1) ]


    def compute(self,input) :
        result = np.array(input)
        for i in range(len(self.ms)) :
            result = sigmoid( np.matmul( result , self.ms[i] ) + self.bs[i] )
        return result[0]


    def mutate(self) :
        rand_pheno = Phenotype()
        clone = Phenotype(self)
        i = choice(range(len(self.ms)))
        if random() > .5 :
            clone.ms[i] += rand_pheno.ms[i]/10.0
        else :
            clone.bs[i] += rand_pheno.bs[i]/10.0
        return clone


    def crossover(self,other) :
        other = Phenotype(other)

        for i in range(len(self.ms)) :
            mask = np.random.rand(*self.ms[i].shape) > 0.5
            other.ms[i] = self.ms[i] * mask + other.ms[i] * (1 - mask)
            
            mask = np.random.rand(*self.bs[i].shape) > 0.5
            other.bs[i] = self.bs[i] * mask + other.bs[i] * (1 - mask)

        return other



# ----------------------------------------------





envir = gym.make('CartPole-v1')



def demo(some=1000,phenotype=Phenotype()):
    state = envir.reset()
    for count in range(some):
        envir.render()
        action = phenotype.compute(state)
        next_state, reward, done, info = envir.step(int(round(action)))
        state = next_state
        if done :
            envir.render()
            break




# -----------------------------------------------

phenotype_population = 50


def mutationFunction(phenotype) :
    if random() > .1 :
        return phenotype.mutate()
    else :
        return Phenotype()

def crossoverFunction(phenotype,other) :
    return phenotype.crossover(other)

def oneTrial(phenotype) :
    state = envir.reset()

    for count in range(501):
        action = int(round( phenotype.compute(state) ))
        state, reward, done, info = envir.step(action)
        if done :
            break;
    return count


def fitnessFunction(phenotype) :
    count = 0.
    total = 0.
    while total < 500 and count < 10 :
        count += 1.
        total += oneTrial(phenotype)
    return 1. * total / count


ga = GeneticAlgorithm(
    mutationFunction = mutationFunction,
    crossoverFunction = crossoverFunction,
    fitnessFunction = fitnessFunction,
    population = [ Phenotype() for i in range(phenotype_population) ],
    populationSize = phenotype_population
    )

try :
    with open('cartpole_best.pickle', 'rb') as handle:
        ga._population[0] = pickle.load(handle)
except :
    pass

print("")
print("-------------------------------------")
print("")
print("Initialize",phenotype_population,"agents for evolution.")
print("Before evolving them the longest time is",int(ga.bestScore()),"frames")
print("")
print("Starting the evolution process")
print("")
try :
    for generation in range(0,501) :
        ga.evolve()
        if generation % 5 == 0:
            print("generation",generation,", best time is",int(ga.bestScore()),"frames")
            demo(phenotype=ga.bestPhenotype())
        if generation % 5 == 2:
            envir.close()
except :
    pass

with open('cartpole_best.pickle', 'wb') as handle:
    pickle.dump(ga.bestPhenotype(),handle)

envir.close()

