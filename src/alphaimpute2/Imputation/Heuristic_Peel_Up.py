import numba
from numba import jit, njit
import numpy as np
import math

from ..tinyhouse import InputOutput
from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import ProbMath

from . import Imputation


try:
    profile
except:
    def profile(x): 
        return x


# Overall thoughts -- this is pretty ugly. Lots of probability math to take lots of things into account. 
# It has the advantage that it *works*. Some of the odd things -- going from genotypes to phased loci and then back again is wierd.
# Right now I do that mostly to keep it consistent with peeling. It may be more straightforward to stay on the genotype plane.
# Alternatively, if we do want to include non-trivial segregation values, it may make sense to keep the full uncertainty.

# Right now things I dislike:
# All of the globals. I wish they were structured better (or mentally structured better in my head)
# All of the for loops with weird index schemes in collapseScoresWithGenotypes and evaluateChild
# Duplication of the expNorm functions -- these are _different_ from AlphaPeel.
# How we get the segregation tensors. Not having to re-calculate is really nice, but maybe comes at a cost?

# Things I like:
# It works.
# There's a lot of good setup here for handling segregation values in a semi-sensible way.
# 

error = 0.01
errorMat = np.array([[1-error, error/2, error/2, error/2], 
                        [error/2, 1-error, 1-error, error/2],
                        [error/2, error/2, error/2, 1-error]], dtype = np.float32)
errorMat = errorMat/np.sum(errorMat, 1)[:,None]
nullError = np.array([.25, .25, .25, .25], dtype = np.float32)

@profile
def singleLocusPeelUp(ind):
    # 5 is probably a pretty reasonable cutoff. Below that it will be hard to impute.
    if len(ind.offspring) > 5:
        nLoci = len(ind.genotypes)
        scores = np.full((4, nLoci), 0, dtype = np.float32)

        for family in ind.families:
            mateScore = np.full((4, 4, nLoci), 0, dtype = np.float32)
            if family.sire is ind:
                mate = family.dam
            if family.dam is ind:
                mate = family.sire

            for child in family.offspring:
                evaluateChild(child.genotypes, mateScore)

            scores += collapseScoresWithGenotypes(mateScore, mate.genotypes)

        newGenotypes = callScore(scores, threshold = 0.99)

        HaplotypeOperations.fillIfMissing(ind.genotypes, newGenotypes)
        HaplotypeOperations.align_individual(ind)

@njit
def collapseScoresWithGenotypes(scores, altGenotypes):
    nLoci = len(altGenotypes)
    # Assume alternative parent is second set of genotypes.
    finalScores = np.full((4, nLoci), 0, dtype = np.float32)

    for i in range(nLoci):
        altGenoProbs = getGenotypeProbabilities(altGenotypes[i])

        # Values is the unnormalized joint probability distribution for parental genotypes.
        values = exp_2D(scores[:, :,i])
        # tmpScore is the unnormalized distribution for the first parents genotypes marginalized over the second.
        tmpScore = np.full(4, 0, dtype = np.float32)
        for j in range(4):
            for k in range(4):
                tmpScore[j] += values[j,k]*altGenoProbs[k]

        for j in range(4):
            finalScores[j, i] = np.log(tmpScore[j] + 1e-8)
    return finalScores

@njit
def getGenotypeProbabilities(genotype) :
    global nullError
    global errorMat

    if genotype == 9:
        return nullError
    else:
        return errorMat[genotype, :]



@njit
def callScore(scores, threshold):
    # Here's the logic here. -- Removed since we just eat the exponential cost and normalize now.
    # We want s1/(s1+s2+s3) > threshold.
    # If we assume s2 > s3, then we can be more restrictive by
    # s1/(s1 + 2s2) > threshold
    # This gives that s1/s2 > 2a/(1-a) where a = threshold.
    # We can log everything to find log(s1) > log(s2) + log(2a/(1-a)). This gives us our cutoff.

    nLoci = scores.shape[1]
    # cutoff = math.log(2*threshold/(1-threshold))
    genotypes = np.full(nLoci, 9, dtype = np.int8)

    # Maybe could do below in a cleaner fasion, but this is nice and explicit.
    for i in range(nLoci) :
        vals = expNorm_1D(scores[:,i])
        s0 = vals[0]
        s1 = vals[1] + vals[2]
        s2 = vals[3]
        if s0 > threshold and s0 > s1 and s0 > s2:
            genotypes[i] = 0

        if s1 > threshold and s1 > s0 and s1 > s2:
            genotypes[i] = 1

        if s2 > threshold and s2 > s0 and s2 > s1:
            genotypes[i] = 2

        # print(scores[:, i], genotypes[i])

    return genotypes
@njit
def evaluateChild(childGenotypes, scores):
    # NOTE: SIRE AND DAM GENOTYPES ARE ARBITRARY CODING.

    nLoci = scores.shape[2]
    for i in range(nLoci):
        genotype = childGenotypes[i]
        if genotype != 9:
            # This is the update -- splitting it into loops for speed.
            segTensor = getLogSegregationForGenotype(genotype)
            for j in range(4):
                for k in range(4):
                    # Assuming no phasing of the child here, and equal segregation.
                        scores[j, k, i] += segTensor[j, k]


segregationTensor = ProbMath.generateSegregation(partial = True) # This will be sire, dam, offspring, although sire + dam are interchangeable.
# Now generate a segregation tensor that goes from genotypes -> phasedGenotypes + error -> probs on parents.
genotypeSegregationTensor = np.einsum("abc, dc -> abd", segregationTensor, errorMat)
logGenotypeSegregationTensor = np.log(genotypeSegregationTensor)

@njit
def getLogSegregationForGenotype(genotype):
    # Splitting this out, since we may want this to be more complicated in the future.
    return logGenotypeSegregationTensor[:, :, genotype]





@jit(nopython=True)
def expNorm_1D(mat):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(4):
        if mat[a] > maxVal or maxVal == 1:
            maxVal = mat[a]
    for a in range(4):
        mat[a] -= maxVal

    # Should flag for better numba-ness.
    tmp = np.exp(mat)
    total = 0
    for a in range(4):
        total += tmp[a]
    for a in range(4):
        tmp[a] /= total
    return tmp


@jit(nopython=True)
def exp_2D(mat):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix (in place).
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(4):
        for b in range(4):
            if mat[a, b] > maxVal or maxVal == 1:
                maxVal = mat[a, b]
    for a in range(4):
        for b in range(4):
            mat[a, b] -= maxVal

    # Should flag for better numba-ness.
    tmp = np.empty(mat.shape, dtype = np.float32)
    for a in range(4):
        for b in range(4):
            tmp[a, b] = np.exp(mat[a, b])
    return tmp





