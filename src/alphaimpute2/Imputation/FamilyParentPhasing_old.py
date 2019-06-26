from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import BasicHMM
import math

import numpy as np
from numba import jit

try:
    profile
except:
    def profile(x): 
        return x

def phaseFounders(pedigree):


    setup_probs() # Set up the global variables. 

    gen0 = pedigree.generations[0].individuals

    targets = [ind for ind in gen0 if len(ind.offspring) > 10]

    for ind in targets:
        phaseIndFromOffspring(ind)


def phaseIndFromOffspring(parent):
    nLoci = len(parent.genotypes)
    nOffspring = len(parent.offspring)
    
    # I don't really want to deal with individuals. I want to deal with haplotypes instead.
    haplotypes = np.array([getHaplotype(ind, parent) for ind in parent.offspring])

    # STEP 2: Assign child positions.
    alignments = np.full((nOffspring, nLoci), 9, dtype = np.int8) # Matrix of 0, 1, and 9
    
    # Set up initial alignments. 

    for i in range(nOffspring):
        # if haplotypes[i, 0] == 0:
        #     alignments[i, 0] = 0
        # elif haplotypes[i, 0] == 1:
        #     alignments[i, 0] = 1
        # else:
        alignments[i, 0] = np.random.randint(2)

    # STEP 3: Create an initial burn in sample, then do a number of passes.


    nSamples = 1
    new_genotypes = np.full((nSamples, nLoci), 1, dtype = np.int8)

    # Burn in run.
    getSample(alignments, parent.peeling_view.genotypeProbabilities, haplotypes, True) 

    forward = False
    for i in range(nSamples):
        print(alignments[:, 990:1000].T)
        new_genotypes[i, :] = getSample(alignments, parent.peeling_view.genotypeProbabilities, haplotypes, forward= False) 

    genotypeProbabilities = np.full((4, nLoci), 0.001, dtype = np.float32)

    for i in range(nLoci):
        for j in range(nSamples):
            genotype = new_genotypes[j, i]
            if genotype != -1:
                genotypeProbabilities[genotype, i] += 1


    genotypeProbabilities /= np.sum(genotypeProbabilities, axis = 0)
    genotypeProbabilities = genotypeProbabilities * (1-.01) + .01/4
    
    print(genotypeProbabilities[:, 0:10].T)

    return genotypeProbabilities
    # parent.peeling_view.penetrance = genotypeProbabilities
    # parent.peeling_view.setGenotypesAll(0.3)


def getHaplotype(ind, parent):
    if ind.sire is parent:
        return ind.haplotypes[0]

    if ind.dam is parent:
        return ind.haplotypes[1]
    return None

@jit(nopython=True)
def getSample(alignments, parent_geno_probs, child_haplotypes, forward):
    nLoci = parent_geno_probs.shape[-1]
    new_genotypes = np.full(nLoci, -1, dtype = np.int8)
    
    scores = np.full(4, 0, dtype = np.float32)


    if forward:
        for i in range(1, nLoci):
            new_genotypes[i] = updateGroups(alignments, i-1, i, parent_geno_probs, child_haplotypes, scores)

    else:
        for i in range(nLoci - 2, -1, -1):
            new_genotypes[i] = updateGroups(alignments, i+1, i, parent_geno_probs, child_haplotypes, scores)
    
    return new_genotypes



@jit(nopython=True)
def updateGroups(alignments, currentLoci, nextLoci, parent_geno_probs, child_haplotypes, scores) :
    global probs_rec
    # Check possible phasings, and correct genotyping errors.
    nOffspring, nLoci = alignments.shape

    scores[:] = 0
    for i in range(nOffspring):
        seg = alignments[i, currentLoci]
        hap = child_haplotypes[i, nextLoci]
        scores += probs_rec[seg, hap, :]

    scores += np.log(parent_geno_probs[:, nextLoci])
    # Randomly sample a value
    
    final_scores = exp_1D_norm(scores)
    genotype, update = callValues(scores, threshold = 0.9)
    # genotype = sample(final_scores)

    for i in range(nOffspring):
        if update:
            alignments[i, nextLoci] = sampleAlignment(alignments[i, currentLoci], child_haplotypes[i, nextLoci], genotype)
        else:
            alignments[i, nextLoci] = alignments[i, currentLoci]
    return genotype

@jit(nopython=True)
def callValues(scores, threshold = 0.9):
    score = scores[0]
    index = 0
    for i in range(1, len(scores)):
        if scores[i] > score:
            score = scores[i]
            index = i

    if score > threshold:
        return index, True
    else:
        return -1, False


@jit(nopython=True)
def sample(scores):
    target = np.random.random()
    index = 0
    while target > scores[index]:
        target -= scores[index]
        index += 1
    return index

@jit(nopython=True)
def sampleAlignment(seg, hap, genotype):
    global probs_no_rec
    rec = 0.00001

    new_seg_0 = probs_no_rec[0, hap, genotype]
    new_seg_1 = probs_no_rec[1, hap, genotype]

    if seg == 0:
        new_seg_0 += np.log(1-rec)
        new_seg_1 += np.log(rec)

    if seg == 1:
        new_seg_0 += np.log(rec)
        new_seg_1 += np.log(1-rec)

    choice = np.random.random()
    p_0 = 1/(1+np.exp(new_seg_1 - new_seg_0))
    if choice < p_0:
        return 0
    else:
        return 1

# @jit(nopython=True)
# def getHetLoci(target):
#     nLoci = len(target)
#     midpoint = math.floor(nLoci/2)
#     locus = -1
#     i = 0
#     while locus < 0 and i < (nLoci/2 + 1):
#         forward = min(nLoci - 1, midpoint + i) 
#         backward = max(0, midpoint - i) 
#         if target[forward] == 1:
#             locus = forward
#         if target[backward] == 1:
#             locus = backward
#         i += 1

#     if locus < 0:
#         locus = midpoint
#     return locus





########## Globals

probs_no_rec = np.full((10, 10, 4), 0, dtype = np.float32)
probs_rec = np.full((10, 10, 4), 0, dtype = np.float32)

def setup_probs(e = 0.01, rec = 0.001):
    global probs_rec
    global probs_no_rec

    for seg in range(10):
        for hap in range(10):
            probs_rec[seg, hap, :] = getProbs_rec(seg, hap, e, rec)
            probs_no_rec[seg, hap, :] = getProbs_no_rec(seg, hap, e = 0)


@jit(nopython=True)
def getProbs_no_rec(seg, hap, e):
    scores = np.full(4, 0, dtype = np.float32)
    
    if hap == 9:
        return scores

    if hap == 0:
        scores[0] = 1-e

        if seg == 0:
            scores[1] = (1-e) 
            scores[2] = e
        if seg == 1:
            scores[1] = e
            scores[2] = 1-e
        if seg == 9:
            scores[1] = .5
            scores[2] = .5
        scores[3] = e

    if hap == 1:
        scores[3] = 1-e
        if seg == 0:
            scores[2] = (1-e) 
            scores[1] = e
        if seg == 1:
            scores[2] = e
            scores[1] = 1-e
        if seg == 9:
            scores[2] = .5
            scores[1] = .5
        scores[0] = e

    return np.log(scores)

@jit(nopython=True)
def getProbs_rec(seg, hap, e, rec):
    scores = np.full(4, 0, dtype = np.float32)
    
    if hap == 9:
        return scores

    if hap == 0:
        scores[0] = 1-e

        if seg == 0:
            scores[1] = (1-rec)*(1-e) + rec*e
            scores[2] = rec*(1-e) + (1-rec)*e
        if seg == 1:
            scores[1] = rec*(1-e) + (1-rec)*e
            scores[2] = (1-rec)*(1-e) + rec*e
        if seg == 9:
            scores[1] = .5
            scores[2] = .5
        scores[3] = e

    if hap == 1:
        scores[3] = 1-e

        if seg == 0:
            scores[2] = (1-rec)*(1-e) + rec*e
            scores[1] = rec*(1-e) + (1-rec)*e
        if seg == 1:
            scores[2] = rec*(1-e) + (1-rec)*e
            scores[1] = (1-rec)*(1-e) + rec*e
        if seg == 9:
            scores[2] = .5
            scores[1] = .5
        scores[0] = e

    return np.log(scores)

########## Utility

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
def norm_1D(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total

