import numpy as np
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





def _sigmoid(x) :
    return 1/(1+np.exp(-x))

def _genNewM(w,h):
    return 4. * ( np.random.rand(w,h) - 0.5 ) / w

def _genNewB(h):
    return ( np.random.rand(h) - 0.5 ) / 5.0


class Phenotype :
    def __init__(self, clone=None, shape=[4,5,5,1], data=None) :

        if clone :
            self.ms = [ np.copy(clone.ms[i]) for i in range(len(clone.ms)) ]
            self.bs = [ np.copy(clone.bs[i]) for i in range(len(clone.bs)) ]
            self.shape = clone.shape
        else :
            self.ms = [ _genNewM(shape[i],shape[i+1]) for i in range(len(shape)-1) ]
            self.bs = [ _genNewB(shape[i+1]) for i in range(len(shape)-1) ]
            self.shape = shape

        if data and shape :
            for i in range(len(self.ms)) :
                self.ms[i] = np.array(data[0][i]).reshape(self.ms[i].shape)
            for i in range(len(self.bs)) :
                self.bs[i] = np.array(data[1][i]).reshape(self.bs[i].shape)


    def clone(self) :
        return Phenotype(self)


    def compute(self,input) :
        result = np.array(input)
        for i in range(len(self.ms)) :
            result = _sigmoid( np.matmul( result , self.ms[i] ) + self.bs[i] )
        return result


    def mutate(self) :
        rand_pheno = Phenotype(shape=self.shape)
        if random() < .05 :
            return rand_pheno
        clone = self.clone()
        i = choice(range(len(self.ms)))
        if random() > .5 :
            clone.ms[i] += rand_pheno.ms[i]/10.0
        else :
            clone.bs[i] += rand_pheno.bs[i]/10.0
        return clone


    def crossover(self,other) :
        other = other.clone()

        for i in range(len(self.ms)) :
            mask = np.random.rand(*self.ms[i].shape) > 0.5
            other.ms[i] = self.ms[i] * mask + other.ms[i] * (1 - mask)
            
            mask = np.random.rand(*self.bs[i].shape) > 0.5
            other.bs[i] = self.bs[i] * mask + other.bs[i] * (1 - mask)

        return other


if __name__ == "__main__" :
    print("Doing a test.  Random")
    a = Phenotype(shape=[4,3,1])
    print("a.ms")
    print(a.ms)
    print("a.ms")
    print(a.ms)
    print("Range")
    a = Phenotype(shape=[4,3,1],data= [ [ range(4*3) , range(3*1) ] , [ range(3), 0 ] ])
    print("a.ms")
    print(a.ms)
    print("a.ms")
    print(a.ms)

