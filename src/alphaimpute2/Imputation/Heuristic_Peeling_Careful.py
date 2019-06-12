import numba
from numba import jit, njit, float32
import numpy as np
import math

from ..tinyhouse import InputOutput
from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import ProbMath

from . import Imputation

np.core.arrayprint._line_width = 200
np.set_printoptions(precision=4, suppress=True, edgeitems=3)


try:
    profile
except:
    def profile(x): 
        return x


# Overall pipeline looks something like:
# From top down:
    # For the pedigree, calculate segregation, and use it to peel down.
#From bottom up:
    # For each parent, peel up offspring.
    # For each parent, re-estimate segregation (with peel-up information).



############
#
#   Peel Down
#
############


def heuristicPeelDown(ind, cutoff = .99):

    # Use the individual's segregation estimate to peel down.

    if ind.sire is not None and ind.dam is not None:
        getAnterior(ind.toJit(), (ind.sire.toJit(), ind.dam.toJit()))

@njit
def getAnterior(ind, parents):

    nLoci = len(ind.genotypes)


    # pat_probs = getTransmittedProbs(ind.segregation[0], parents[0].genotypeProbabilities)
    # mat_probs = getTransmittedProbs(ind.segregation[1], parents[1].genotypeProbabilities)
    pat_probs = getTransmittedProbs_raw(ind.raw_segregation[0], parents[0].genotypeProbabilities)
    mat_probs = getTransmittedProbs_raw(ind.raw_segregation[1], parents[1].genotypeProbabilities)

    e = 0.00001
    for i in range(nLoci):
        pat = pat_probs[i] + e/2
        mat = mat_probs[i] + e/2

        ind.anterior[0,i] = (1-pat)*(1-mat)
        ind.anterior[1,i] = (1-pat)*mat
        ind.anterior[2,i] = pat*(1-mat)
        ind.anterior[3,i] = pat*mat



@njit
def getTransmittedProbs_raw(seg, genoProbs):
    nLoci = len(seg)
    probs = np.full(nLoci, .5, np.float32)

    for i in range(nLoci):
        p = genoProbs[:, i]
        
        # tmp_raw = .5
        # if seg[i] < 0.01:
        #     tmp_raw = 0
        # elif seg[i] > 0.99 and seg[i] <= 1:
        #     tmp_raw = 1
        # else:
        #     tmp_raw = .5
        
        tmp = seg[i]

        p_1 = tmp*p[1] + (1-tmp)*p[2] + p[3]
        probs[i] = p_1
    return probs


@njit
def getTransmittedProbs(seg, genoProbs):
    nLoci = len(seg)
    probs = np.full(nLoci, .5, np.float32)

    for i in range(nLoci):
        p = genoProbs[:, i]
        if seg[i] == 0:
            p_1 = p[2] + p[3]
        if seg[i] == 1:
            p_1 = p[1] + p[3]
        if seg[i] == 9:
            p_1 = .5*p[1] + .5*p[2] + p[3]
        probs[i] = p_1
    return probs

@njit
def fillInFromParentAndSeg(segregation, haplotype, parent):
    nLoci = len(segregation)
    for i in range(nLoci):
        if haplotype[i] == 9:
            if parent.genotypes[i] == 0:
                haplotype[i] = 0
            elif parent.genotypes[i] == 2:
                haplotype[i] = 1

            if segregation[i] == 0:
                haplotype[i] = parent.haplotypes[0][i]
            if segregation[i] == 1:
                haplotype[i] = parent.haplotypes[1][i]

############
#
#   Peel Up
#
############


def HeuristicPeelUp(ind, cutoff = 0.99):

    # Use an individual's segregation estimate to peel up and reconstruct the individual based on their offspring's genotypes.
    if len(ind.offspring) > 1:
        nLoci = len(ind.genotypes)

        # Scores represent genotype probabilities. 
        #Our goal is to generate join scores for the sire and dam for each parent, and then join them together for a single score for a single parent.
        scores = np.full((4, nLoci), 0, dtype = np.float32)

        for family in ind.families:
            # For each family we want to generate join genotype distributions for the sire and dam.

            mateScore = np.full((4, 4, nLoci), 0, dtype = np.float32)
            if family.sire is ind:
                mate = family.dam
                mateIsSire = False
            if family.dam is ind:
                mate = family.sire
                mateIsSire = True

            # We peel the child up to both of their parents.
            for child in family.offspring:
                child.setGenotypesPosterior(cutoff)
                evaluateChild(child.toJit(), mateScore)

            # This is to make this work for sire or dams. Just transposing the matrix so the main individual is always the "sire".
            if mateIsSire :
                mateScore = np.transpose(mateScore, [1, 0, 2])

            # We then collapse the scores by marginalizing over their mate's genotype.
            mate.setGenotypesAll(cutoff)
            scores += collapseScoresWithGenotypes(mateScore, mate.toJit())

        # We then see if we can call the scores.
        # ind.clearGenotypes()

        set_posterior_from_scores(scores, ind.posterior)
        # HaplotypeOperations.align_individual(ind)

        # ind.toJit().setValueFromGenotypes(ind.posterior)

# Need to change genotype proabilities to handle phasing.

@njit
def evaluateChild(child, scores):
    # NOTE: SIRE AND DAM GENOTYPES ARE ARBITRARY CODING.

    nLoci = scores.shape[2]
    for i in range(nLoci):
        if child.genotypes[i] != 9 or child.haplotypes[0][i] !=9 or child.haplotypes[1][i] !=9:
            segTensor = getLogSegregationForGenotype(child, i)
            for j in range(4):
                for k in range(4):
                    # Assuming no phasing of the child here, and equal segregation.
                    scores[j, k, i] += segTensor[j, k]




# This sets up how you turn haplotypes + genotypes into genotype probabilities. 
# Indexing is hap0, hap1, geno, GenotypeProbabilities.
# We just use this to marginalize over the dam's genotype. Maybe there are better ways to do this.

def generateGenoProbs():
    global geno_probs
    error = .01
    geno_probs[9,9,9] = np.array([.25, .25, .25, .25], dtype = np.float32) * (1-error) + error/4
    geno_probs[9,9,1] = np.array([0, .5, .5, 0], dtype = np.float32) * (1-error) + error/4

    geno_probs[0,9,9] = np.array([.5, .5, 0, 0], dtype = np.float32) * (1-error) + error/4
    geno_probs[1,9,9] = np.array([0, 0, .5, .5], dtype = np.float32) * (1-error) + error/4
    geno_probs[9,0,9] = np.array([.5, 0, .5, 0], dtype = np.float32) * (1-error) + error/4
    geno_probs[9,1,9] = np.array([0, .5, 0, .5], dtype = np.float32) * (1-error) + error/4

    geno_probs[1,0,1] = np.array([0, 0, 1, 0], dtype = np.float32) * (1-error) + error/4
    geno_probs[0,1,1] = np.array([0, 1, 0, 0], dtype = np.float32) * (1-error) + error/4
    geno_probs[0,0,0] = np.array([1, 0, 0, 0], dtype = np.float32) * (1-error) + error/4
    geno_probs[1,1,2] = np.array([0, 0, 0, 1], dtype = np.float32) * (1-error) + error/4

geno_probs = np.full((10, 10, 10, 4), .25, dtype = np.float32) # Because 9 indexing for missing.
generateGenoProbs()

@njit
def getGenotypeProbabilities(ind, i) :
    global geno_probs
    return geno_probs[ind.haplotypes[0][i], ind.haplotypes[1][i], ind.genotypes[i], :]


# Now generate a segregation tensor that goes from genotypes -> phasedGenotypes + error -> probs on parents.
# logGenotypeSegregationTensor = np.log(genotypeSegregationTensor)

def generateLogGenotypeSegregationTensor():
    error = .01
    global logGenotypeSegregationTensor
    global genotypeSegregationTensor
    segregationTensor = ProbMath.generateSegregation() # This will be sire, dam, offspring, although sire + dam are interchangeable.

    seg_probs = np.full((10, 10, 4), .25, dtype = np.float32)
    seg_probs[0,0] = np.array([1, 0, 0, 0], dtype = np.float32) * (1-error) + error/4
    seg_probs[0,1] = np.array([0, 1, 0, 0], dtype = np.float32) * (1-error) + error/4
    seg_probs[1,0] = np.array([0, 0, 1, 0], dtype = np.float32) * (1-error) + error/4
    seg_probs[1,1] = np.array([0, 0, 0, 1], dtype = np.float32) * (1-error) + error/4

    seg_probs[0,9] = np.array([.5, .5, 0, 0], dtype = np.float32) * (1-error) + error/4
    seg_probs[1,9] = np.array([0, 0, .5, .5], dtype = np.float32) * (1-error) + error/4

    seg_probs[9,0] = np.array([.5, 0, .5, 0], dtype = np.float32) * (1-error) + error/4
    seg_probs[9,1] = np.array([0, .5, 0, .5], dtype = np.float32) * (1-error) + error/4

    seg_probs[9,9] = np.array([.25, .25, .25, .25], dtype = np.float32) * (1-error) + error/4

    genotypeSegregationTensor = np.full((10, 10, 10, 10, 10, 4, 4), .25, dtype = np.float32)

    for seg0 in [0, 1, 9]:
        for seg1 in [0, 1, 9]:
            for hap0 in [0, 1, 9]:
                for hap1 in [0, 1, 9]:
                    for geno in [0, 1, 2, 9]:
                        genotypes = geno_probs[hap0, hap1, geno]
                        segregation = seg_probs[seg0, seg1]

                        genotypeSegregationTensor[seg0, seg1, hap0, hap1, geno,:, :] = np.einsum("abcd, c, d -> ab", segregationTensor, genotypes, segregation)
                        logGenotypeSegregationTensor = np.log(genotypeSegregationTensor)
genotypeSegregationTensor = None
logGenotypeSegregationTensor = None
generateLogGenotypeSegregationTensor()

@njit
def getLogSegregationForGenotype(child, i):

    # Basically we want to be able to go from seg[0] + seg[1] + genotype + hap[0] + hap[1] => joint parental genotype.
    seg0 = child.segregation[0][i]
    seg1 = child.segregation[1][i]
    # seg0 = child.segregation[0][i]
    # seg1 = child.segregation[1][i]

    hap0 = child.haplotypes[0][i]
    hap1 = child.haplotypes[1][i]

    geno = child.genotypes[i]
    return logGenotypeSegregationTensor[seg0, seg1, hap0, hap1, geno]



@njit
def collapseScoresWithGenotypes(scores, mate):

    nLoci = len(mate.genotypes)
    # Assume alternative parent is second set of genotypes.
    finalScores = np.full((4, nLoci), 0, dtype = np.float32)

    for i in range(nLoci):
        # For each loci, get the genotype probability, then marginalize.
        # Could multithread.
        altGenoProbs = getGenotypeProbabilities(mate, i)
        logMarginalize(scores[:,:,i], altGenoProbs, finalScores[:,i])
    return finalScores

@njit
def logMarginalize(scores, altGenoProbs, finalScores):
        # Values is the unnormalized joint probability distribution for parental genotypes.
        values = exp_2D(scores)
        # tmpScore is the unnormalized distribution for the first parents genotypes marginalized over the second.
        tmpScore = np.full(4, 0, dtype = np.float32)
        for j in range(4):
            for k in range(4):
                tmpScore[j] += values[j,k]*altGenoProbs[k]

        for j in range(4):
            # The 1e-8 represents a small genotype uncertainty term.
            finalScores[j] = np.log(tmpScore[j] + 1e-8)

@njit
def callScore(scores, posterior, threshold):
    nLoci = scores.shape[1]
    posterior[:,:] = 1

    # Maybe could do below in a cleaner fasion, but this is nice and explicit.
    for i in range(nLoci) :
        vals = expNorm_1D(scores[:,i])

        for j in range(4):
            if vals[j] < 1-threshold :
                posterior[j, i] = 0

        if np.sum(posterior[:,i]) == 0:
            print(posterior[:,i])
            print(scores[:,i])
@njit
def set_posterior_from_scores(scores, posterior):
    nLoci = scores.shape[1]
    posterior[:,:] = 1
    e = 0.001
    # Maybe could do below in a cleaner fasion, but this is nice and explicit.
    for i in range(nLoci) :
        vals = expNorm_1D(scores[:,i])

        for j in range(4):
            posterior[j, i] = vals[j]*(1-e) + e/4           
# @njit
# def callScore(scores, threshold, ind):
#     # I am going to assume threshold > .5.
#     # Maybe skip loci where we know we are already good?
#     nLoci = scores.shape[1]

#     # Maybe could do below in a cleaner fasion, but this is nice and explicit.
#     for i in range(nLoci) :
#         vals = expNorm_1D(scores[:,i])
#         vals = combineAndNorm(vals, getGenotypeProbabilities(ind, i))
#         s0 = vals[0]
#         s1 = vals[1] + vals[2]
#         s2 = vals[3]

#         # Set genotypes.
#         if s0 > threshold:
#             setIfMissing(ind.genotypes, 0, i)
#         if s1 > threshold:
#             setIfMissing(ind.genotypes, 1, i)
#         if s2 > threshold:
#             setIfMissing(ind.genotypes, 2, i)

        
#         # Paternal Haplotypes
#         if vals[0] + vals[1] > threshold :
#                 setIfMissing(ind.haplotypes[0], 0, i)
#         if vals[2] + vals[3] > threshold :
#                 setIfMissing(ind.haplotypes[0], 1, i)
        
#         # Maternal Haplotypes
#         if vals[0] + vals[2] > threshold :
#                 setIfMissing(ind.haplotypes[1], 0, i)
#         if vals[1] + vals[3] > threshold :
#                 setIfMissing(ind.haplotypes[1], 1, i)



@njit
def setIfMissing(mat, val, i):
    if mat[i] == 9:
        mat[i] = val

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

@njit
def norm(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total

@njit
def combineAndNorm(mat1, mat2):
    for i in range(len(mat1)):
        mat1[i] *= mat2[i]
    norm(mat1)
    return(mat1)

@jit(nopython=True)
def exp_2D(mat):
    # Matrix is 4x4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix (in place).
    # Question: Does explicit dimensionality help or hinder?
    # Question: Could we also make this work for nLoci as well?
    # i.e. can we stick these all somewhere together and not replicate?
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

############
#
#   Estimate Segregation
#
############

def setSegregation(ind, cutoff = 0.99):

    # Roughly the idea is something like:
    # Grab the individual's genotype (or whatever we're using).
    # Determine how this could align to the parental haplotypes (in some sense these are independent).
    if ind.sire is not None and ind.dam is not None:
        nLoci = len(ind.genotypes)
        pointEstimates = np.full((4, nLoci), 1, dtype = np.float32)

        # Set genotypes for individual and parents.
        ind.setGenotypesPosterior(cutoff)
        ind.sire.setGenotypesAll(cutoff)
        ind.dam.setGenotypesAll(cutoff)

        fillPointEstimates(pointEstimates, ind.toJit(), ind.sire.toJit(), ind.dam.toJit())

        # Run the smoothing algorithm on this.
        smoothedEstimates = smoothPointSeg(pointEstimates, 1.0/nLoci)

        # Then call the segregation values.
        callSegregation(ind.segregation, ind.raw_segregation, smoothedEstimates, cutoff = 0.99)

@njit
def callSegregation(segregation, raw_segregation, estimate, cutoff):
    nLoci = len(segregation[0])
    for i in range(nLoci):

        paternalSeg = estimate[2, i] + estimate[3,i]
        maternalSeg = estimate[1, i] + estimate[3,i]

        raw_segregation[0][i] = paternalSeg
        raw_segregation[1][i] = maternalSeg

        if paternalSeg > cutoff:
            segregation[0][i] = 1
        elif paternalSeg < 1 - cutoff:
            segregation[0][i] = 0
        else:
            segregation[0][i] = 9

        if maternalSeg > cutoff:
            segregation[1][i] = 1
        elif maternalSeg < 1 - cutoff:
            segregation[1][i] = 0
        else:
            segregation[1][i] = 9
@njit
def fillPointEstimates(pointEstimates, ind, sire, dam):
    nLoci = pointEstimates.shape[1]
    e = 0.01
    for i in range(nLoci):
        # Let's do sire side.
        # I'm going to assume we've already peeled down.
        if ind.haplotypes[0][i] != 9:
            indhap = ind.haplotypes[0][i]
            sirehap0 = sire.haplotypes[0][i]
            sirehap1 = sire.haplotypes[1][i]
            if indhap == sirehap0 and indhap != sirehap1 and sirehap1 != 9:
                pointEstimates[0,i] *= 1-e
                pointEstimates[1,i] *= 1-e
                pointEstimates[2,i] *= e
                pointEstimates[3,i] *= e
            
            if indhap == sirehap1 and indhap != sirehap0 and sirehap0 != 9:
                pointEstimates[0,i] *= e
                pointEstimates[1,i] *= e
                pointEstimates[2,i] *= 1-e
                pointEstimates[3,i] *= 1-e

            # This can be super problematic
            # if indhap == sirehap0 and indhap != sirehap1 and sirehap1 == 9:
            #     pointEstimates[0,i] *= 1-e
            #     pointEstimates[1,i] *= 1-e
            #     pointEstimates[2,i] *= .5
            #     pointEstimates[3,i] *= .5

            # if indhap == sirehap1 and indhap != sirehap0 and sirehap1 == 9:
            #     pointEstimates[0,i] *= .5
            #     pointEstimates[1,i] *= .5
            #     pointEstimates[2,i] *= 1-e
            #     pointEstimates[3,i] *= 1-e

        if ind.haplotypes[1][i] != 9:
            indhap = ind.haplotypes[1][i]
            damhap0 = dam.haplotypes[0][i]
            damhap1 = dam.haplotypes[1][i]
            if indhap == damhap0 and indhap != damhap1 and damhap1 != 9:
                pointEstimates[0,i] *= 1-e
                pointEstimates[2,i] *= 1-e
                pointEstimates[1,i] *= e
                pointEstimates[3,i] *= e
            
            if indhap == damhap1 and indhap != damhap0 and damhap0 != 9:
                pointEstimates[0,i] *= e
                pointEstimates[2,i] *= e
                pointEstimates[1,i] *= 1-e
                pointEstimates[3,i] *= 1-e
            # This can be super problematic
            # if indhap == damhap0 and indhap != damhap1 and damhap1 == 9:
            #     pointEstimates[0,i] *= 1-e
            #     pointEstimates[2,i] *= 1-e
            #     pointEstimates[1,i] *= .5
            #     pointEstimates[3,i] *= .5

            # if indhap == damhap1 and indhap != damhap0 and damhap1 == 9:
            #     pointEstimates[0,i] *= .5
            #     pointEstimates[2,i] *= .5
            #     pointEstimates[1,i] *= 1-e
            #     pointEstimates[3,i] *= 1-e





@jit(nopython=True, nogil=True, locals={'e': float32, 'e2':float32, 'e1e':float32, 'e2i':float32})
def smoothPointSeg(pointSeg, transmission):

    # This is the forward backward algorithm.
    # Segregation estimate state ordering: pp, pm, mp, mm
    nLoci = pointSeg.shape[1] 

    seg = np.full(pointSeg.shape, .25, dtype = np.float32)
    for i in range(nLoci):
        for j in range(4):
            seg[j,i] = pointSeg[j,i]

    tmp = np.full((4), 0, dtype = np.float32)
    new = np.full((4), 0, dtype = np.float32)

    prev = np.full((4), .25, dtype = np.float32)
    e = transmission
    e2 = e**2
    e1e = e*(1-e)
    e2i = (1.0-e)**2

    for i in range(1, nLoci):
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i-1]
        
        sum_j = 0
        for j in range(4):
            sum_j += tmp[j]
        for j in range(4):
            tmp[j] = tmp[j]/sum_j

        # !                  fm  fm  fm  fm 
        # !segregationOrder: pp, pm, mp, mm

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        # tmp = tmp/np.sum(tmp)
        # new = e2i*tmp + e2 + e1e*(tmp[0] + tmp[3])*same + e1e*(tmp[1] + tmp[2])*diff       

        for j in range(4):
            seg[j,i] *= new[j]
        # seg[:,i] *= new
        prev = new

    prev = np.full((4), .25, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i+1]
        
        sum_j = 0
        for j in range(4):
            sum_j += tmp[j]
        for j in range(4):
            tmp[j] = tmp[j]/sum_j

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[j,i] *= new[j]
        prev = new
    
    for i in range(nLoci):
        sum_j = 0
        for j in range(4):
            sum_j += seg[j, i]
        for j in range(4):
            seg[j, i] = seg[j, i]/sum_j

    # seg = seg/np.sum(seg, 0)
    return(seg)

