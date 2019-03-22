from random import choice, random, shuffle


def defaultMutation(phenotype) : return phenotype
def defaultCrossover(a,b) : return a
def defaultFitness(phenotype) : return 0


class GeneticAlgorithm:

    def __init__(self,mutationFunction=defaultMutation,
        crossoverFunction=defaultCrossover,fitnessFunction=defaultFitness,
        doesABeatBFunction=None,population=None,populationSize=100):

        self._mutationFunction = mutationFunction
        self._crossoverFunction = crossoverFunction
        self._fitnessFunction = fitnessFunction
        self._doesABeatBFunction = doesABeatBFunction
        self._population = population if population else []
        self._populationSize = populationSize

        if len(self._population) <= 0 :
            assert("population must be an array and contain at least 1 phenotypes")
        if self._populationSize <= 0 :
            assert("populationSize must be greater than 0")

        self._scoredPopulation = None

    def _populate(self) :
        size = len(self._population)
        while len(self._population) < self._populationSize :
            self._population.append( self._mutate(choice(self._population)) )

    def _mutate(self, phenotype) :
        return self._mutationFunction(phenotype)

    def _crossover(self, phenotype) :
        return self._crossoverFunction( phenotype, choice(self._population) )

    def _doesABeatB(self, a, b) :
        if self._doesABeatBFunction :
            return self._doesABeatBFunction(a,b)
        else :
            return self._fitnessFunction(a) > self._fitnessFunction(b)


    def _compete(self) :
        nextGeneration = []

        for index in range(0,len(self._population),2) :
            phenotype = self._population[index]
            competitor = self._population[index+1]

            nextGeneration.append(phenotype)
            if self._doesABeatB( phenotype , competitor ) :
                if random() < 0.5 :
                    nextGeneration.append(self._mutate(phenotype))
                else :
                    nextGeneration.append(self._crossover(phenotype))
            else :
                nextGeneration.append(competitor)

        self._population = nextGeneration


    def _randomizePopulationOrder(self) :
        shuffle(self._population)


    def evolve(self) :
        self._scoredPopulation = None
        self._populate()
        self._randomizePopulationOrder()
        self._compete()
        return self


    def scoredPopulation(self) :
        if not self._scoredPopulation :
            self._scoredPopulation = [ [phenotype, self._fitnessFunction(phenotype)] for phenotype in self._population]
            self._scoredPopulation.sort(key=lambda a:-a[1])
        return self._scoredPopulation


    def bestPhenotype(self) :
        scored = self.scoredPopulation()
        return scored[0][0]

    def bestScore(self) :
        scored = self.scoredPopulation()
        return scored[0][1]

    def meanScore(self) :
        scores = [ item[1] for item in self.scoredPopulation() ]
        return int(1.0 * sum(scores) / len(scores))
