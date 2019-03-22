import numpy as np
from random import choice, random

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def genNewM(w,h):
    return 4. * ( np.random.rand(w,h) - 0.5 ) / w

def genNewB(h):
    return ( np.random.rand(h) - 0.5 ) / 5.0


class Phenotype :
    def __init__(self, clone=None) :

        if clone :
            self.ms = [ np.copy(clone.ms[i]) for i in range(len(clone.ms)) ]
            self.bs = [ np.copy(clone.bs[i]) for i in range(len(clone.bs)) ]
        else :
            self.ms = [ genNewM(4,5) , genNewM(5,5) , genNewM(5,1) ]
            self.bs = [ genNewB(5) ,   genNewB(5) ,   genNewB(1) ]


    def clone(self) :
        return Phenotype(self)


    def compute(self,input) :
        result = np.array(input)
        for i in range(len(self.ms)) :
            result = sigmoid( np.matmul( result , self.ms[i] ) + self.bs[i] )
        return result[0]


    def mutate(self) :
        rand_pheno = Phenotype()
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

