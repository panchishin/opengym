from geneticalgorithm import GeneticAlgorithm
import math
from random import random

phenotype_size = 10
phenotype_population = 100


def mutationFunction(phenotype) :
    gene = int(math.floor( random() * phenotype_size ))
    phenotype = list(phenotype)
    phenotype[gene] += random() * random() * 20 - 10
    return phenotype


def crossoverFunction(a, b) :
    x = list(a)
    y = list(b)

    for i in range(phenotype_size) :
        if random() > 0.5 :
            x[i] = b[i]
            y[i] = a[i]

    return x , y


def fitnessFunction(phenotype) :
    # assume perfect solution is '50.0' for all numbers
    return -1.0 * sum([ math.pow( 50.0 - i , 2 ) for i in phenotype ])


def createRandomPhenotype() :
    return [ random()*100 for i in range(phenotype_size) ]


ga = GeneticAlgorithm(
    mutationFunction = mutationFunction,
    crossoverFunction = crossoverFunction,
    fitnessFunction = fitnessFunction,
    population = [ createRandomPhenotype() for i in range(phenotype_population) ],
    populationSize = phenotype_population
    )


print("The starting score is:",ga.bestScore())
for iteration in range(1,1001) :
    ga.evolve()
    if iteration % 100 == 0:
        print("iteration",iteration,", best score =",ga.bestScore())

