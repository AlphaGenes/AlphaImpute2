import numba
from numba import jit, njit, float32
import numpy as np
import math

from ..tinyhouse import InputOutput
from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import ProbMath

from . import Imputation

import concurrent.futures
from itertools import repeat


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

# Error terms:
# Peel down: Assume 0.00001 genotyping/mutation error for anterior terms.
# Peel up: assume 1% in genotypes when converting genotypes => probabilities
# Peel up: assume 1% error in segregation when converting called seg => probabilities.
# Peel up: assume 1% error when going from mate genotype probs => genotype probs.
# Peel up: 1e-8 error rate when marginalizing across mate genotypes (shouldn't need this with the mate genotype probs?)
# Peel up: additional 0.001 error rate when going from posterior -> genotype probabilities.
# Segregation: assumes 1% genotyping error when creating match between individual + parent.

# We use called genotypes for posterior + set segregation.


@profile
def runHeuristicPeeling(pedigree, args):
    # Set penetrance values
    for ind in pedigree:
        ind.toJit().setValueFromGenotypes(ind.penetrance)

    # Set anterior values for founders. These will never change, so only need to do this once.
    pedigree.setMaf()
    founder_anterior = ProbMath.getGenotypesFromMaf(pedigree.maf)
    founder_anterior = founder_anterior*(1-0.1) + 0.1/4 # I want to add a bit of noise here so we aren't fixing genotypes without a good reason.
    for ind in pedigree:
        if ind.isFounder():
            ind.setAnterior(founder_anterior.copy())

    # Run peeling cycle.
    cutoffs =       [.99, .9, .9, .9, .9]
    write_cutoffs = [.99, .9, .7, .6, .3]
    for cycle in range(len(cutoffs)):
        print("Imputation cycle ", cycle)

        pedigreePeelDown(pedigree, args, cutoffs[cycle])
        pedigreePeelUp(pedigree, args, cutoffs[cycle])

    for ind in pedigree:
        ind.jit_view.setGenotypesAll(.3)

    pedigree.writeGenotypes(args.out + ".genotypes." + str(4))

@profile
def pedigreePeelDown(pedigree, args, cutoff):

    if args.maxthreads == 1:
        for generation in pedigree.generations:
            for parent in generation.parents:
                parent.jit_view.setGenotypesAll(cutoff)

            for ind in generation.individuals:
                ind.jit_view.setGenotypesPosterior(cutoff)

            for ind in generation.individuals:
                setSegregation(ind)
                heuristicPeelDown(ind)
    else:
        for generation in pedigree.generations:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.maxthreads) as executor:
                executor.map(lambda ind, cutoff: ind.jit_view.setGenotypesAll(cutoff), generation.parents, repeat(cutoff))
                executor.map(lambda ind, cutoff: ind.jit_view.setGenotypesPosterior(cutoff), generation.individuals, repeat(cutoff))
                executor.map(setSegregation, generation.individuals)
                executor.map(heuristicPeelDown, generation.individuals)


# if args.maxthreads > 1:
#     with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
#          results = executor.map(Peeling.peel, jit_families, repeat(Peeling.PEEL_DOWN), repeat(peelingInfo), repeat(singleLocusMode))
# else:
#     for family in jit_families:
#         Peeling.peel(family, Peeling.PEEL_DOWN, peelingInfo, singleLocusMode)

@profile
def pedigreePeelUp(pedigree, args, cutoff):
    if args.maxthreads == 1:

        for generation in reversed(pedigree.generations):
            for parent in generation.parents:
                parent.jit_view.setGenotypesAll(cutoff)

            for ind in generation.individuals:
                ind.setPosteriorFromNew()
                ind.jit_view.setGenotypesPosterior(cutoff)

            for ind in generation.individuals:
                setSegregation(ind)

            for family in generation.families:
                heuristicPeelUp_family(family)
    else:
        for generation in pedigree.generations:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.maxthreads) as executor:
                # Set parents
                executor.map(lambda ind, cutoff: ind.jit_view.setGenotypesAll(cutoff), generation.parents, repeat(cutoff))

                # Re-align offspring
                executor.map(lambda ind: ind.setPosteriorFromNew(), generation.individuals)
                executor.map(lambda ind, cutoff: ind.jit_view.setGenotypesPosterior(cutoff), generation.individuals, repeat(cutoff))
                executor.map(setSegregation, generation.individuals)

                # Run per-family peeling.
                executor.map(heuristicPeelUp_family, generation.families)




############
#
#   Peel Down
#
############


@profile
def heuristicPeelDown(ind):

    # Use the individual's segregation estimate to peel down.
    if ind.sire is not None and ind.dam is not None:
        newAnterior = getAnterior(ind.toJit(), (ind.sire.toJit(), ind.dam.toJit()))
        ind.setAnterior(newAnterior)

@jit(nopython=True, nogil = True)
def getAnterior(ind, parents):

    nLoci = len(ind.genotypes)

    anterior = np.full((4, nLoci), 0, dtype = np.float32)

    pat_probs = getTransmittedProbs(ind.segregation[0], parents[0].genotypeProbabilities)
    mat_probs = getTransmittedProbs(ind.segregation[1], parents[1].genotypeProbabilities)

    #Small error in genotypes
    e = 0.00001
    for i in range(nLoci):
        pat = pat_probs[i]*(1-e) + e/2
        mat = mat_probs[i]*(1-e) + e/2

        anterior[0,i] = (1-pat)*(1-mat)
        anterior[1,i] = (1-pat)*mat
        anterior[2,i] = pat*(1-mat)
        anterior[3,i] = pat*mat
    return anterior



@jit(nopython=True, nogil = True)
def getTransmittedProbs(seg, genoProbs):
    # For each loci, calculate the probability of the parent transmitting a 1 allele.
    # This will use the child's segregation value, and the parent's genotype probabilities.

    nLoci = len(seg)
    probs = np.full(nLoci, .5, np.float32)

    for i in range(nLoci):
        p = genoProbs[:, i]
        p_1 = seg[i]*p[1] + (1-seg[i])*p[2] + p[3]
        probs[i] = p_1
    return probs


############
#
#   Peel Up
#
############


@profile
def heuristicPeelUp_family(fam):

    # Use an individual's segregation estimate to peel up and reconstruct the individual based on their offspring's genotypes.
    nLoci = len(fam.sire.genotypes)

    # Scores represent the log genotype probabilities across all mates
    sire_scores = np.full((4, nLoci), 0, dtype = np.float32)
    dam_scores = np.full((4, nLoci), 0, dtype = np.float32)

    combined_score = np.full((4, 4, nLoci), 0, dtype = np.float32)

    # We peel the child up to both of their parents.
    for child in fam.offspring:
        evaluateChild(child.toJit(), combined_score)

    # set the sire scores.
    sire_scores = collapseScoresWithGenotypes(combined_score, fam.dam.toJit())
    fam.sire.addPosterior(sire_scores, fam.idn)

    # Rotate the matrix to get the dam scores.
    combined_score = np.transpose(combined_score, [1, 0, 2])

    dam_scores = collapseScoresWithGenotypes(combined_score, fam.sire.toJit())
    fam.dam.addPosterior(dam_scores, fam.idn)


@jit(nopython=True, nogil = True)
def evaluateChild(child, scores):
    nLoci = scores.shape[2]
    for i in range(nLoci):
        if child.genotypes[i] != 9 or child.haplotypes[0][i] !=9 or child.haplotypes[1][i] !=9:
            segTensor = getLogSegregationForGenotype(child, i)
            for j in range(4):
                for k in range(4):
                    scores[j, k, i] += segTensor[j, k]

@jit(nopython=True, nogil = True)
def getLogSegregationForGenotype(child, i):

    # Basically we want to be able to go from seg[0] + seg[1] + genotype + hap[0] + hap[1] => joint parental genotype.
    
    threshold = 0.99

    seg0 = convert_seg_to_int(child.segregation[0][i], threshold)
    seg1 = convert_seg_to_int(child.segregation[1][i], threshold)

    hap0 = child.haplotypes[0][i]
    hap1 = child.haplotypes[1][i]

    geno = child.genotypes[i]
    return logGenotypeSegregationTensor[seg0, seg1, hap0, hap1, geno]

@jit(nopython=True, nogil = True)
def convert_seg_to_int(val, threshold):
    if val < 1-threshold:
        return 0
    if val > threshold:
        return 1
    return 9


@jit(nopython=True, nogil = True)
def collapseScoresWithGenotypes(scores, mate):
    nLoci = len(mate.genotypes)
    # Assume alternative parent is second set of genotypes.
    finalScores = np.full((4, nLoci), 0, dtype = np.float32)

    e = 0.001

    for i in range(nLoci):
        # For each loci, get the genotype probability, then marginalize.
        altGenoProbs = mate.genotypeProbabilities[:,i].copy()
        for j in range(4):
            altGenoProbs[j] = altGenoProbs[j]*(1-e) + e/4


        # altGenoProbs = getGenotypeProbabilities(mate, i)
        # diff = 0
        # for j in range(4):
        #     diff += np.abs(altGenoProbs[j] - tmp[j])
        # if diff > 0.05:
        #     print(altGenoProbs, getGenotypeProbabilities(mate, i), mate.genotypes[i], mate.haplotypes[0][i], mate.haplotypes[1][i])

        logMarginalize(scores[:,:,i], altGenoProbs, finalScores[:,i])
    return finalScores

@jit(nopython=True, nogil = True)
def logMarginalize(scores, altGenoProbs, finalScores):
    # Values is the unnormalized joint probability distribution for parental genotypes.
    values = exp_2D_norm(scores)
    # tmpScore is the unnormalized distribution for the first parents genotypes marginalized over the second.
    tmpScore = np.full(4, 0, dtype = np.float32)
    for j in range(4):
        for k in range(4):
            tmpScore[j] += values[j,k]*altGenoProbs[k]

    for j in range(4):
        tmpScore[j] += 1e-8

    norm_1D(tmpScore)
    for j in range(4):
        # The 1e-8 represents a small genotype uncertainty term.
        finalScores[j] = np.log(tmpScore[j])



@jit(nopython=True, nogil = True)
def set_posterior_from_scores(scores):
    nLoci = scores.shape[1]
    posterior = np.full((4, nLoci), 1, dtype = np.float32)
    e = 0.001
    # Maybe could do below in a cleaner fasion, but this is nice and explicit.
    for i in range(nLoci) :
        vals = exp_1D_norm(scores[:,i])
        
        for j in range(4):
            posterior[j, i] = vals[j]*(1-e) + e/4           
    return posterior

############
#
#   Estimate Segregation
#
############

@profile
def setSegregation(ind):

    # Roughly the idea is something like:
    # Grab the individual's genotype (or whatever we're using).
    # Determine how this could align to the parental haplotypes (in some sense these are independent).
    if ind.sire is not None and ind.dam is not None:
        nLoci = len(ind.genotypes)

        # # Set genotypes for individual and parents.
        
        pointEstimates = np.full((4, nLoci), 1, dtype = np.float32)
        fillPointEstimates(pointEstimates, ind.toJit(), ind.sire.toJit(), ind.dam.toJit())

        # Run the smoothing algorithm on this.
        smoothedEstimates = smoothPointSeg(pointEstimates, 1.0/nLoci) # This is where different map lengths could be added.

        # Then call the segregation values.
        callSegregation(ind.segregation, smoothedEstimates)

@jit(nopython=True, nogil = True)
def callSegregation(segregation, estimate):
    nLoci = len(segregation[0])
    for i in range(nLoci):

        paternalSeg = estimate[2, i] + estimate[3,i]
        maternalSeg = estimate[1, i] + estimate[3,i]

        segregation[0][i] = paternalSeg
        segregation[1][i] = maternalSeg

@jit(nopython=True, nogil = True)
def fillPointEstimates(pointEstimates, ind, sire, dam):
    nLoci = pointEstimates.shape[1]
    e = 0.01 # Assume 1% genotyping error.
    for i in range(nLoci):
        # Let's do sire side.
        # I'm going to assume we've already peeled down.
        if ind.haplotypes[0][i] != 9:
            indhap = ind.haplotypes[0][i]
            sirehap0 = sire.haplotypes[0][i]
            sirehap1 = sire.haplotypes[1][i]

            # If both parental haplotypes are non-missing, and
            # not equal to each other, then this is an informative marker.
            if sirehap0 != 9 and sirehap1 != 9 and sirehap0 != sirehap1:

                if indhap == sirehap0 :
                    pointEstimates[0,i] *= 1-e
                    pointEstimates[1,i] *= 1-e
                    pointEstimates[2,i] *= e
                    pointEstimates[3,i] *= e

                if indhap == sirehap1:
                    pointEstimates[0,i] *= e
                    pointEstimates[1,i] *= e
                    pointEstimates[2,i] *= 1-e
                    pointEstimates[3,i] *= 1-e

        if ind.haplotypes[1][i] != 9:
            indhap = ind.haplotypes[1][i]
            damhap0 = dam.haplotypes[0][i]
            damhap1 = dam.haplotypes[1][i]

            if damhap0 != 9 and damhap1 != 9 and damhap0 != damhap1:

                if indhap == damhap0:
                    pointEstimates[0,i] *= 1-e
                    pointEstimates[2,i] *= 1-e
                    pointEstimates[1,i] *= e
                    pointEstimates[3,i] *= e
                
                if indhap == damhap1 :
                    pointEstimates[0,i] *= e
                    pointEstimates[2,i] *= e
                    pointEstimates[1,i] *= 1-e
                    pointEstimates[3,i] *= 1-e

@jit(nopython=True, nogil=True, locals={'e': float32, 'e2':float32, 'e1e':float32, 'e2i':float32})
def smoothPointSeg(pointSeg, transmission):

    nLoci = pointSeg.shape[1] 

    # Seg is the output, and is a copy of pointseg.
    seg = np.full(pointSeg.shape, .25, dtype = np.float32)
    for i in range(nLoci):
        for j in range(4):
            seg[j,i] = pointSeg[j,i]

    # Variables.
    tmp = np.full(4, 0, dtype = np.float32)
    new = np.full(4, 0, dtype = np.float32)
    prev = np.full(4, .25, dtype = np.float32)

    # Transmission constants.
    e = transmission
    e2 = e**2
    e1e = e*(1-e)
    e2i = (1.0-e)**2

    for i in range(1, nLoci):
        # Combine previous estimate with previous pointseg and then transmit forward.
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i-1]
        
        norm_1D(tmp)

        # Father/Mother; Paternal/Maternal
        # !                  fm  fm  fm  fm 
        # !segregationOrder: pp, pm, mp, mm

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[j,i] *= new[j]
        prev = new

    prev = np.full((4), .25, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i+1]
        
        norm_1D(tmp)

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[j,i] *= new[j]
        prev = new
    
    for i in range(nLoci):
        norm_1D(seg[:,i])

    return(seg)


############
#
#  Numba matrix functions
#
###########

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
def exp_2D_norm(mat):
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
    tmp = np.full(mat.shape, 0, dtype = np.float32)
    for a in range(4):
        for b in range(4):
            tmp[a, b] = np.exp(mat[a, b])


    # Normalize.
    score = 0
    for a in range(4):
        for b in range(4):
            score += tmp[a, b]
    for a in range(4):
        for b in range(4):
            tmp[a, b]/=score
    return tmp

############
#
#   STATIC METHODS + GLOBALS
#
############

# This sets up how you turn haplotypes + genotypes into genotype probabilities. 
# Indexing is hap0, hap1, geno, GenotypeProbabilities.

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
