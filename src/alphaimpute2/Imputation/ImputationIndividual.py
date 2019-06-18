from ..tinyhouse import Pedigree

import numba
from numba import jit, int8, int64, boolean, deferred_type, optional, jitclass, float32, double
from collections import OrderedDict
import numpy as np
from . import Heuristic_Peeling_Careful

try:
    profile
except:
    def profile(x): 
        return x


class AlphaImputeIndividual(Pedigree.Individual):
    def __init__(self, idx, idn):
        super().__init__(idx, idn)

        self.segregation = None
        self.originalGenotypes = None
        self.anterior = None
        self.penetrance = None
        self.posterior = None

        self.newPosterior = None


    def setupIndividual(self):
        nLoci = len(self.genotypes)
        self.segregation = (np.full(nLoci, .5, dtype = np.float32), np.full(nLoci, .5, dtype = np.float32))
        self.originalGenotypes = self.genotypes.copy()

        self.anterior = np.full((4, nLoci), 1, dtype = np.float32) # There's probably some space saving format we could do here... 
        self.penetrance = np.full((4, nLoci), 1, dtype = np.float32) 
        self.genotypeProbabilities = np.full((4, nLoci), 1, dtype = np.float32)


        if len(self.offspring) > 0:
            self.posterior = np.full((4, nLoci), 1, dtype = np.float32)
            self.has_offspring = True
        else:
            self.posterior = np.full((0, 0), 1, dtype = np.float32)
            self.has_offspring = False


        self.setJit()

    def setJit(self):
        # Set the 
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError("In order to just in time an Individual, both genotypes and haplotypes need to be not None")
        nLoci = len(self.genotypes) # self.genotypes will always be not None (otherwise error will be raised above).

        self.jit_view = jit_Individual(self.idn, self.genotypes, self.haplotypes, self.segregation, self.anterior, self.penetrance, self.posterior, self.genotypeProbabilities, self.has_offspring, nLoci)


    def toJit(self):
        return self.jit_view

    def setAnterior(self, newAnterior):
        self.anterior = newAnterior
        self.jit_view.anterior = self.anterior
        self.jit_view.currentState = -1

    def setPosterior(self, newPosterior):
        self.posterior = newPosterior
        self.jit_view.posterior = self.posterior
        self.jit_view.currentState = -1

    def setPosteriorFromNew(self):
        if self.newPosterior is not None:
            # Take all the posterior values and add them up.
            sumPosterior = None
            for post in self.newPosterior.values():
                if sumPosterior is None:
                    sumPosterior = post
                else:
                    sumPosterior += post

            self.posterior = Heuristic_Peeling_Careful.set_posterior_from_scores(sumPosterior)
            self.jit_view.posterior = self.posterior
            self.jit_view.currentState = -1
            self.newPosterior = None


    def addPosterior(self, newValues, idn):
        if self.newPosterior is None:
            self.newPosterior = dict()
        self.newPosterior[idn] = newValues

spec = OrderedDict()
spec['idn'] = int64
spec['nLoci'] = int64
spec['genotypes'] = int8[:]

spec['has_offspring'] = boolean
# Haplotypes and reads are a tuple of int8 and int64.
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

spec['segregation'] = numba.typeof((np.array([0, 1], dtype = np.float32), np.array([0], dtype = np.float32)))

spec['anterior'] = float32[:,:]
spec['penetrance'] = float32[:,:]
spec['posterior'] = float32[:,:]
spec['genotypeProbabilities'] = float32[:,:]

spec['currentState'] = int8
spec['currentCutoff'] = float32



@jitclass(spec)
class jit_Individual(object):
    def __init__(self, idn, genotypes, haplotypes, segregation, anterior, penetrance, posterior, genotypeProbabilities, has_offspring, nLoci):
        self.idn = idn

        self.genotypes = genotypes
        self.haplotypes = haplotypes
        self.segregation = segregation

        self.anterior = anterior
        self.penetrance = penetrance
        self.posterior = posterior

        self.has_offspring = has_offspring

        self.genotypeProbabilities = genotypeProbabilities

        self.nLoci = nLoci

        self.currentState = -1
        self.currentCutoff = 0

    def check_and_set_state(self, state, cutoff):
        if self.currentState != state or self.currentCutoff != cutoff:
            self.currentState = state
            self.currentCutoff = cutoff
            return True
        else:
            return False

    def setGenotypesAll(self, cutoff = 0.99):
        if self.check_and_set_state(0, cutoff):
            self.setGenotypesFromPeelingData(True, True, True, cutoff)

    def setGenotypesPosterior(self, cutoff = 0.99):
        if self.check_and_set_state(1, cutoff):
            self.setGenotypesFromPeelingData(False, True, True, cutoff)

    def setGenotypesPenetrance(self, cutoff = 0.99):
        if self.check_and_set_state(2, cutoff):
            self.setGenotypesFromPeelingData(False, True, False, cutoff)


    def setGenotypesAnterior(self, cutoff = 0.99):
        if self.check_and_set_state(3, cutoff):
            self.setGenotypesFromPeelingData(True, True, False, cutoff)




    def setValueFromGenotypes(self, mat):
        nLoci = self.nLoci
        mat[:,:] = 1
        for i in range(nLoci):
            g = self.genotypes[i]
            
            if g == 0:
                mat[0,i] = 1
                mat[1,i] = 0
                mat[2,i] = 0
                mat[3,i] = 0
            if g == 1:
                mat[0,i] = 0
                mat[1,i] = 1
                mat[2,i] = 1
                mat[3,i] = 0

            if g == 2:
                mat[0,i] = 0
                mat[1,i] = 0
                mat[2,i] = 0
                mat[3,i] = 1

            # Handle haplotypes by rulling out genotype states
            if self.haplotypes[0][i] == 0:
                mat[2,i] = 0
                mat[3,i] = 0

            if self.haplotypes[0][i] == 1:
                mat[0,i] = 0
                mat[1,i] = 0

            if self.haplotypes[1][i] == 0:
                mat[1,i] = 0
                mat[3,i] = 0

            if self.haplotypes[1][i] == 1:
                mat[0,i] = 0
                mat[2,i] = 0

            e = 0.01
            count = 0
            for j in range(4):
                count += mat[j, i]
            for j in range(4):
                mat[j, i] = mat[j, i]/count*(1-e) + e/4

    def setGenotypesFromPeelingData(self, useAnterior = False, usePenetrance = False, usePosterior = False, cutoff = 0.99):
        setGenotypesFromPeelingData_ngil(self, useAnterior, usePenetrance, usePosterior, cutoff)
    #     nLoci = self.nLoci

    #     self.genotypeProbabilities[:,:] = 1
    #     finalGenotypes = self.genotypeProbabilities
    #     if useAnterior:
    #         finalGenotypes *= self.anterior
    #     if usePosterior and self.has_offspring:
    #         finalGenotypes *= self.posterior
    #     if usePenetrance:
    #         finalGenotypes *= self.penetrance

    #     self.normalize(finalGenotypes)

    #     # set genotypes/haplotypes from this value.
    #     for i in range(nLoci):
    #         # We now set to missing below.
    #         # self.genotypes[i] = 9
    #         # self.haplotypes[0][i] = 9
    #         # self.haplotypes[1][i] = 9


    #         # Set genotype.
    #         maxGenotype = 0
    #         maxVal = finalGenotypes[0,i]

    #         if finalGenotypes[1,i] + finalGenotypes[2,i] > maxVal:
    #             maxGenotype = 1
    #             maxVal = finalGenotypes[1,i] + finalGenotypes[2,i] 
    #         if finalGenotypes[3,i] > maxVal:
    #             maxGenotype = 2
    #             maxVal = finalGenotypes[3,i]

    #         if maxVal > cutoff:
    #             self.genotypes[i] = maxGenotype
    #         else: 
    #             self.genotypes[i] = 9


    #         # Set haplotype.

    #         hap0 = finalGenotypes[2, i] + finalGenotypes[3, i]
    #         hap1 = finalGenotypes[1, i] + finalGenotypes[3, i]

    #         if hap0 > cutoff:
    #             self.haplotypes[0][i] = 1
    #         elif hap0 < 1 - cutoff:
    #             self.haplotypes[0][i] = 0
    #         else:
    #             self.haplotypes[0][i] = 9


    #         if hap1 > cutoff:
    #             self.haplotypes[1][i] = 1
    #         elif hap1 < 1 - cutoff:
    #             self.haplotypes[1][i] = 0
    #         else:
    #             self.haplotypes[1][i] = 9

@jit(nopython=True, nogil = True)
def normalize(values):

    for i in range(values.shape[1]):
        count = 0
        for j in range(4):
            count += values[j, i]

        for j in range(4):
            if count != 0:
                values[j, i]/= count
            else:
                values[j, i] = .25


@jit(nopython=True, nogil = True)
def setGenotypesFromPeelingData_ngil(ind, useAnterior = False, usePenetrance = False, usePosterior = False, cutoff = 0.99):
    nLoci = ind.nLoci

    ind.genotypeProbabilities[:,:] = 1
    finalGenotypes = ind.genotypeProbabilities
    if useAnterior:
        finalGenotypes *= ind.anterior
    if usePosterior and ind.has_offspring:
        finalGenotypes *= ind.posterior
    if usePenetrance:
        finalGenotypes *= ind.penetrance

    normalize(finalGenotypes)

    # set genotypes/haplotypes from this value.
    for i in range(nLoci):
        # We now set to missing below.
        # self.genotypes[i] = 9
        # self.haplotypes[0][i] = 9
        # self.haplotypes[1][i] = 9


        # Set genotype.
        maxGenotype = 0
        maxVal = finalGenotypes[0,i]

        if finalGenotypes[1,i] + finalGenotypes[2,i] > maxVal:
            maxGenotype = 1
            maxVal = finalGenotypes[1,i] + finalGenotypes[2,i] 
        if finalGenotypes[3,i] > maxVal:
            maxGenotype = 2
            maxVal = finalGenotypes[3,i]

        if maxVal > cutoff:
            ind.genotypes[i] = maxGenotype
        else: 
            ind.genotypes[i] = 9


        # Set haplotype.

        hap0 = finalGenotypes[2, i] + finalGenotypes[3, i]
        hap1 = finalGenotypes[1, i] + finalGenotypes[3, i]

        if hap0 > cutoff:
            ind.haplotypes[0][i] = 1
        elif hap0 < 1 - cutoff:
            ind.haplotypes[0][i] = 0
        else:
            ind.haplotypes[0][i] = 9


        if hap1 > cutoff:
            ind.haplotypes[1][i] = 1
        elif hap1 < 1 - cutoff:
            ind.haplotypes[1][i] = 0
        else:
            ind.haplotypes[1][i] = 9




