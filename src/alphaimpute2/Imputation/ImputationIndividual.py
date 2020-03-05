from ..tinyhouse import Pedigree

import numba
from numba import jit, int8, int64, boolean, optional, jitclass, float32
from collections import OrderedDict
import numpy as np
from . import Imputation

try:
    profile
except:
    def profile(x): 
        return x


class AlphaImputeIndividual(Pedigree.Individual):
    def __init__(self, idx, idn):
        super().__init__(idx, idn)

        self.reverse_view = None

    def setupIndividual(self):


        nLoci = len(self.genotypes)
        if self.haplotypes is None:
            self.haplotypes = (np.full(nLoci, 9, dtype = np.int8), np.full(nLoci, 9, dtype = np.int8))

        # self.setPhasingView()
        self.setPeelingView()

    def setPeelingView(self):
        # Set the 
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError("In order to create a jit_Peeling_Individual both the genotypes and haplotypes need to be created.")
        nLoci = len(self.genotypes) # self.genotypes will always be not None (otherwise error will be raised above).

        if len(self.offspring) > 0:
            self.has_offspring = True
        else:
            self.has_offspring = False

        self.peeling_view = jit_Peeling_Individual(self.idn, self.genotypes, self.haplotypes, self.has_offspring, nLoci)

    def setPhasingView(self):
        # Set the 
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError("In order to create a jit_Phasing_Individual both the genotypes and haplotypes need to be created.")
        nLoci = len(self.genotypes) # self.genotypes will always be not None (otherwise error will be raised above).

        self.phasing_view = jit_Phasing_Individual(self.idn, self.genotypes, self.haplotypes, nLoci)


    def reverse_individual(self):
        new_ind = AlphaImputeIndividual(self.idx, self.idn)
        new_ind.genotypes = np.ascontiguousarray(np.flip(self.genotypes))

        new_ind.setupIndividual()
        Imputation.ind_align(new_ind)

        self.reverse_view = new_ind
        new_ind.reverse_view = self
        return(new_ind)

    def add_backward_info(self):
        if self.reverse_view is None:
            print("Trying to set backward information, but no reverse_view is availible")
        else:
            self.phasing_view.backward[:,:] = np.flip(self.reverse_view.phasing_view.forward, axis = 1) # Flip along loci.



spec = OrderedDict()
spec['idn'] = int64
spec['nLoci'] = int64
spec['genotypes'] = int8[:]

spec['called_genotypes'] = int8[:]

# Haplotypes and reads are a tuple of int8 and int64.
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))
spec['current_haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

spec['penetrance'] = float32[:,:]

spec['forward'] = float32[:,:]
spec['backward'] = float32[:,:]

spec['own_haplotypes'] = int64[:,:]
spec['has_own_haplotypes'] = boolean

@jitclass(spec)
class jit_Phasing_Individual(object):
    '''
    This class holds data for phasing a given individual.
    '''
    def __init__(self, idn, genotypes, haplotypes, nLoci):
        self.idn = idn
        self.nLoci = nLoci
        self.genotypes = genotypes
        self.haplotypes = haplotypes
        self.current_haplotypes = (haplotypes[0].copy(), haplotypes[1].copy())
        
        self.penetrance = np.full((4, nLoci), 1, dtype = np.float32) 

        self.forward = np.full((4, nLoci), 1, dtype = np.float32) 
        self.backward = np.full((4, nLoci), 1, dtype = np.float32) 

        self.called_genotypes = np.full(nLoci, 9, dtype = np.int8)

        self.own_haplotypes = np.full((0, 0), 0, dtype = np.int64)
        self.has_own_haplotypes = False

    def set_own_haplotypes(self, haplotypes_in):
        self.own_haplotypes = haplotypes_in
        self.has_own_haplotypes = True

example_phasing_individual = None
def get_example_phasing_individual():
    global example_phasing_individual
    if example_phasing_individual is None:
        example_phasing_individual = jit_Phasing_Individual(-1, np.array([0, 1], dtype = np.int8), (np.array([0, 1], dtype = np.int8), np.array([0,1], dtype = np.int8)), 2)
    return example_phasing_individual

spec = OrderedDict()
spec['idn'] = int64
spec['nLoci'] = int64
spec['genotypes'] = int8[:]

spec['has_offspring'] = boolean
# Haplotypes and reads are a tuple of int8 and int64.
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

spec['segregation'] = numba.typeof((np.array([0, 1], dtype = np.float32), np.array([0], dtype = np.float32)))
spec['posterior_staging'] = optional(numba.typeof([np.full((4, 100), 0, dtype = np.float32)]))

spec['anterior'] = float32[:,:]
spec['penetrance'] = float32[:,:]
spec['posterior'] = float32[:,:]

spec['newPosterior'] = float32[:,:]

spec['genotypeProbabilities'] = float32[:,:]

spec['currentState'] = int8
spec['currentCutoff'] = float32



@jitclass(spec)
class jit_Peeling_Individual(object):
    '''
    This class holds a lot of arrays for peeling individuals and a handful of functions to translate genotypes => probabilities and back again.
    '''
    def __init__(self, idn, genotypes, haplotypes, has_offspring, nLoci):
        self.nLoci = nLoci
        self.idn = idn
        self.genotypes = genotypes
        self.haplotypes = haplotypes
        


        # Initial value for segregation is .5 to represent uncertainty between haplotype inheritance.
        self.segregation = (np.full(nLoci, .5, dtype = np.float32), np.full(nLoci, .5, dtype = np.float32))

        # Create the posterior terms.
        self.has_offspring = has_offspring
        # self.has_offspring = True
        if self.has_offspring:
            self.posterior = np.full((4, nLoci), 1, dtype = np.float32)
            self.anterior = np.full((4, nLoci), 1, dtype = np.float32) 
            self.penetrance = np.full((4, nLoci), 1, dtype = np.float32) 

            self.newPosterior = np.full((4, nLoci), 0, dtype = np.float32) 
            self.genotypeProbabilities = np.full((4, nLoci), 1, dtype = np.float32)

        else:
            self.fill_in_haplotypes()

            self.posterior = np.full((0, 0), 1, dtype = np.float32)
            self.anterior = np.full((0, 0), 1, dtype = np.float32) 
            self.penetrance = np.full((0, 0), 1, dtype = np.float32) 
            self.genotypeProbabilities = np.full((0, 0), 1, dtype = np.float32)

        self.posterior_staging = None

        # Current state indicates which genotype combination an individual is set to, and the cutoff value used for calling their genotypes.
        # Missing = -1
        # ALL = 0
        # POSTERIOR = 1
        # PENTRANCE = 2
        # Anterior = 3
        # We only ever use ALL or POSTERIOR.

        self.currentState = -1
        self.currentCutoff = 0

    def check_and_set_state(self, state, cutoff):
        if not self.has_offspring:
            return False
        
        if self.currentState != state or self.currentCutoff != cutoff:
            self.currentState = state
            self.currentCutoff = cutoff
            return True
        else:
            return False

    def fill_in_haplotypes(self):
        for i in range(self.nLoci):
            if self.genotypes[i] == 0:
                
                if self.haplotypes[0][i] == 9:
                    self.haplotypes[0][i] = 0
                
                if self.haplotypes[1][i] == 9:
                    self.haplotypes[1][i] = 0

            if self.genotypes[i] == 2:
                
                if self.haplotypes[0][i] == 9:
                    self.haplotypes[0][i] = 1
                
                if self.haplotypes[1][i] == 9:
                    self.haplotypes[1][i] = 1

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


    def setAnterior(self, newAnterior):
        self.anterior = newAnterior
        if self.currentState == 0 or self.currentState == 3:
            # i.e. if current state is ALL or ANTERIOR which use the anterior estimate.
            # If the current state is not one of those two, we will re-calculate the genotype probabilities when we set the individual to one of those states.
            self.currentState = -1

    def setPosterior(self):
        if self.has_offspring:
            self.popPosterior()
            self.currentState = -1
            self.posterior = self.set_posterior_from_scores(self.newPosterior)
            self.newPosterior[:,:] = 0

    def popPosterior(self):
        if self.has_offspring and self.posterior_staging is not None:
            # Take all the posterior values and add them up.
            nPost = len(self.posterior_staging)
            for i in range(nPost):
                self.newPosterior += self.posterior_staging[i]
            self.posterior_staging = None

    def addPosterior(self, newValues, idn):
        if self.posterior_staging is None:
            self.posterior_staging = [newValues]
        else:
            self.posterior_staging.append(newValues)


    def setupProbabilityValues(self, error_rate):
        if self.has_offspring: # Only need to do this for founders.
            self.setValueFromGenotypes(self.penetrance, error_rate) 

    def setValueFromGenotypes(self, mat, error_rate):
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

            e = error_rate
            count = 0
            for j in range(4):
                count += mat[j, i]
            # if count == 0:
            #     print(g, self.haplotypes[0][i], self.haplotypes[1][i])
            for j in range(4):
                mat[j, i] = mat[j, i]/count*(1-e) + e/4

    def setGenotypesFromPeelingData(self, useAnterior = False, usePenetrance = False, usePosterior = False, cutoff = 0.99):
        # setGenotypesFromPeelingData_ngil(self, useAnterior, usePenetrance, usePosterior, cutoff)
        nLoci = self.nLoci

        self.genotypeProbabilities[:,:] = 1
        finalGenotypes = self.genotypeProbabilities
        if useAnterior:
            finalGenotypes *= self.anterior
        
        if usePosterior and self.has_offspring:
            finalGenotypes *= self.posterior
        
        if usePenetrance:
            finalGenotypes *= self.penetrance

        self.setGenotypesFromGenotypeProbabilities(finalGenotypes, cutoff)

    def setGenotypesFromGenotypeProbabilities(self, finalGenotypes, cutoff):
        nLoci = self.nLoci
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
                self.genotypes[i] = maxGenotype
            else: 
                self.genotypes[i] = 9


            # Set haplotype.

            hap0 = finalGenotypes[2, i] + finalGenotypes[3, i]
            hap1 = finalGenotypes[1, i] + finalGenotypes[3, i]

            if hap0 > cutoff:
                self.haplotypes[0][i] = 1
            elif hap0 < 1 - cutoff:
                self.haplotypes[0][i] = 0
            else:
                self.haplotypes[0][i] = 9


            if hap1 > cutoff:
                self.haplotypes[1][i] = 1
            elif hap1 < 1 - cutoff:
                self.haplotypes[1][i] = 0
            else:
                self.haplotypes[1][i] = 9

    def set_posterior_from_scores(self, scores):
        nLoci = scores.shape[1]
        posterior = np.full((4, nLoci), 1, dtype = np.float32)
        e = 0.001
        # Maybe could do below in a cleaner fasion, but this is nice and explicit.
        for i in range(nLoci) :
            vals = exp_1D_norm(scores[:,i])
            
            for j in range(4):
                posterior[j, i] = vals[j]*(1-e) + e/4           
        return posterior



@jit(nopython=True, nogil = True)
def norm_1D(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total


@jit(nopython=True, nogil = True)
def exp_1D_norm(mat):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(4):
        if mat[a] > maxVal or maxVal == 1:
            maxVal = mat[a]
    for a in range(4):
        mat[a] -= maxVal

    # Should flag for better numba-ness.
    tmp = np.full(4, 0, dtype = np.float32)
    for a in range(4):
        tmp[a] = np.exp(mat[a])

    norm_1D(tmp)

    return tmp



@jit(nopython=True, nogil = True)
def normalize(values):
    # Normalize values along a single axis.
    for i in range(values.shape[1]):
        count = 0
        for j in range(4):
            count += values[j, i]

        for j in range(4):
            if count != 0:
                values[j, i]/= count
            else:
                values[j, i] = .25
