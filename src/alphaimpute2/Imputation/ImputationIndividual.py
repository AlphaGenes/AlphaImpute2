from ..tinyhouse import Pedigree

import numba
from numba import jit, int8, int64, boolean, deferred_type, optional, jitclass, float32, double
from collections import OrderedDict
import numpy as np


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

        self.currentState = None

    def setupIndividual(self):
        nLoci = len(self.genotypes)
        self.segregation = (np.full(nLoci, 9, dtype = np.int8), np.full(nLoci, 9, dtype = np.int8))
        self.raw_segregation = (np.full(nLoci, .5, dtype = np.float32), np.full(nLoci, .5, dtype = np.float32))
        self.originalGenotypes = self.genotypes.copy()

        self.anterior = np.full((4, nLoci), 1, dtype = np.float32) # There's probably some space saving format we could do here... 
        self.penetrance = np.full((4, nLoci), 1, dtype = np.float32) 
        self.posterior = np.full((4, nLoci), 1, dtype = np.float32)
        self.genotypeProbabilities = np.full((4, nLoci), 1, dtype = np.float32)

        self.setJit()

    def setJit(self):
        # Set the 
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError("In order to just in time an Individual, both genotypes and haplotypes need to be not None")
        nLoci = len(self.genotypes) # self.genotypes will always be not None (otherwise error will be raised above).

        self.jit_view = jit_Individual(self.idn, self.genotypes, self.haplotypes, self.segregation, self.anterior, self.penetrance, self.posterior, self.genotypeProbabilities, self.raw_segregation, nLoci)


    def toJit(self):
        return self.jit_view


    def setGenotypesPenetrance(self, cutoff = 0.99):
        # if self.currentState != "penetrance":
        #     self.currentState = "penetrance"
        self.jit_view.setGenotypesFromPeelingData(False, True, False, cutoff)

    def setGenotypesAll(self, cutoff = 0.99):
        # if self.currentState != "all":
        #     self.currentState = "all"
        self.jit_view.setGenotypesFromPeelingData(True, True, True, cutoff)

    def setGenotypesAnterior(self, cutoff = 0.99):
        # if self.currentState != "anterior" :
        #     self.currentState = "anterior" 
        self.jit_view.setGenotypesFromPeelingData(True, True, False, cutoff)

    def setGenotypesPosterior(self, cutoff = 0.99):
        # if self.currentState != "posterior":
        #     self.currentState = "posterior"
        self.jit_view.setGenotypesFromPeelingData(False, True, True, cutoff)

    def clearGenotypes(self):
        self.genotypes[:] = 9
        self.haplotypes[0][:] = 9
        self.haplotypes[1][:] = 9


    def setAnterior(self, newAnterior):
        self.anterior = newAnterior
        self.jit.anterior = self.anterior
        self.currentState = None

    def setPosterior(self, newPosterior):
        self.posterior = newPosterior
        self.jit.posterior = self.posterior
        self.currentState = None


spec = OrderedDict()
spec['idn'] = int64
spec['nLoci'] = int64
spec['genotypes'] = int8[:]
# Haplotypes and reads are a tuple of int8 and int64.
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))
spec['segregation'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

spec['raw_segregation'] = numba.typeof((np.array([0, 1], dtype = np.float32), np.array([0], dtype = np.float32)))

spec['anterior'] = float32[:,:]
spec['penetrance'] = float32[:,:]
spec['posterior'] = float32[:,:]
spec['genotypeProbabilities'] = float32[:,:]



@jitclass(spec)
class jit_Individual(object):
    def __init__(self, idn, genotypes, haplotypes, segregation, anterior, penetrance, posterior, genotypeProbabilities, raw_segregation, nLoci):
        self.idn = idn

        self.genotypes = genotypes
        self.haplotypes = haplotypes
        self.segregation = segregation

        self.anterior = anterior
        self.penetrance = penetrance
        self.posterior = posterior

        self.genotypeProbabilities = genotypeProbabilities

        self.raw_segregation = raw_segregation

        self.nLoci = nLoci

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
        nLoci = self.nLoci

        finalGenotypes = np.full((4, nLoci), 1, dtype = np.float32) 
        if useAnterior:
            finalGenotypes *= self.anterior
        if usePosterior:
            finalGenotypes *= self.posterior
        if usePenetrance:
            finalGenotypes *= self.penetrance


        for i in range(nLoci):
            for j in range(4):
                if finalGenotypes[j, i] < -0.0001:
                    print(self.anterior[:, i], self.posterior[:, i], self.penetrance[:, i], finalGenotypes[:, i]) 


        self.normalize(finalGenotypes)
        self.genotypeProbabilities[:,:] = finalGenotypes[:,:]

        # finalGenotypes /= np.sum(finalGenotypes, 1)
        # set genotypes/haplotypes from this value.
        for i in range(nLoci):
            self.genotypes[i] = 9
            self.haplotypes[0][i] = 9
            self.haplotypes[1][i] = 9


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
                self.genotypes[i] = maxGenotype

            # Set haplotype.

            hap0 = finalGenotypes[2, i] + finalGenotypes[3, i]
            hap1 = finalGenotypes[1, i] + finalGenotypes[3, i]

            if hap0 > cutoff:
                self.haplotypes[0][i] = 1
            elif hap0 < 1 - cutoff:
                self.haplotypes[0][i] = 0

            if hap1 > cutoff:
                self.haplotypes[1][i] = 1
            elif hap1 < 1 - cutoff:
                self.haplotypes[1][i] = 0

    def normalize(self, values):

        for i in range(values.shape[1]):
            count = 0
            for j in range(4):
                count += values[j, i]

            for j in range(4):
                if count != 0:
                    values[j, i]/= count
                else:
                    values[j, i] = .25



