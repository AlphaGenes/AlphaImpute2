from numba import jit, int8, int32, boolean, jitclass, float32, int64
import numpy as np

from . import Imputation



    #####################################
    #####################################
    ####                            #####
    ####      Pedigree Programs     #####
    ####                            #####
    #####################################
    #####################################




def performPeeling(individuals, fill = 0, ancestors = False, peelUp = False):
    for ind in individuals:
        performPeeling_ind(ind, fill, ancestors)

def performPeeling_ind(ind, fill, ancestors = False, peelUp = False):
    parentHomozygoticFillIn(ind)
    ind_peelDown_prob(ind, threshold = fill, peelUp = peelUp)
    if ancestors:
        # Imputation.imputeFromAncestors(ind, depth = 0, cutoff = fill) #This functions in a similar manner to the line before.
        imputeFromAncestors(ind, depth = 1, cutoff = fill)
        imputeFromAncestors(ind, depth = 2, cutoff = fill)


    #####################################
    #####################################
    ####                            #####
    ####     Pedigree Imputation    #####
    ####                            #####
    #####################################
    #####################################




def parentHomozygoticFillIn(ind) :
    if ind.sire is not None:
        Imputation.ind_fillInGenotypesFromPhase(ind.sire)
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[0], ind.sire.genotypes)

    if ind.dam is not None:
        Imputation.ind_fillInGenotypesFromPhase(ind.dam)
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[1], ind.dam.genotypes)

    Imputation.fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])


    #####################################
    #####################################
    ####                            #####
    ####  ProbPedigree Imputation   #####
    ####                            #####
    #####################################
    #####################################


def ind_peelDown_prob(ind, threshold = .99, peelUp = False):

    #If they have a sire, they have to have a dam.
    if ind.sire is not None and ind.dam is not None:
        peelDown_prob(ind.haplotypes, ind.sire.haplotypes, ind.dam.haplotypes, threshold = threshold, peelUp = peelUp)
        Imputation.ind_align(ind)
@jit(nopython=True)
def peelDown_prob(haps, sireHaps, damHaps, threshold, peelUp = False):
    # 1) create pointSeg 
    # 2) do a forward-backward on the pentrance
    # 3) find loci that have a clear segregation, and call them.
    pointSeg = getPeelDownPointSeg(haps, sireHaps, damHaps)
    nLoci = len(haps[0])

    rate = 1.0/nLoci
    values = loopyPeelingForwardBack(pointSeg, rate)
    # print(values)
    #Now set values.
    for parIndex in range(2):
        if parIndex == 0:
            parHaps = sireHaps
        else:
            parHaps = damHaps
        callChildHaps(haps[parIndex], parIndex, values, parHaps, threshold)
        
    #Currently not using this. Need a good argument for a use case I think.
    if peelUp:
        for parIndex in range(2):
            if parIndex == 0:
                parHaps = sireHaps
            else:
                parHaps = damHaps
            calledHaps = callParentHaps(haps[parIndex], parIndex, values, parHaps, threshold)

            Imputation.filInIfMissing(parHaps[0], calledHaps[0])
            Imputation.filInIfMissing(parHaps[1], calledHaps[1])


@jit(nopython=True)
def callChildHaps(hap, parIndex, values, parHaps, threshold):
    nLoci = len(hap)
    thresh0 = 1-threshold
    thresh1 = threshold
    for i in range(nLoci):
        if hap[i] == 9:
            if parIndex == 0:
                #Val is probablity of hap 1
                val = values[i,2] + values[i,3]
            else:
                val = values[i,1] + values[i,3]
            # print(thresh0, val)
            if val < thresh0:
                hap[i] = parHaps[0][i]
            elif val > thresh1:
                hap[i] = parHaps[1][i]

@jit(nopython=True)
def callParentHaps(hap, parIndex, values, parHaps, threshold):
    nLoci = len(hap)
    thresh0 = 1-threshold
    thresh1 = threshold
    calledHaps = np.full((nLoci, 2), 9, dtype = np.int8)

    for i in range(nLoci):
        if hap[i] != 9:
            if parIndex == 0:
                #Val is probablity of hap 1
                val = values[i,2] + values[i,3]
            else:
                val = values[i,1] + values[i,3]
            if val < thresh0:
                if parHaps[0][i] == 9:
                    calledHaps[0][i] = hap[i]

            elif val > thresh1:
                if parHaps[1][i] == 9:
                    calledHaps[1][i] = hap[i]                
    return calledHaps

@jit(nopython=True)
def getPeelDownPointSeg(haps, sireHaps, damHaps):

    e = 0.01
    ei = 1-e
    #Same as peeling (I think).
    # pp, pm, mp, mm
    nLoci = len(haps[0])
    pointSeg = np.full((nLoci, 4), 1, dtype = np.float32)

    for parIndex in range(2):
        if parIndex == 0:
            parHaps = sireHaps
        else:
            parHaps = damHaps
        for i in range(nLoci):
            #Not hugely sure what to do if one is missing and the other is not.
            #Will see if this is a problem later.
            if haps[parIndex][i] != 9 and parHaps[0][i] != 9 and parHaps[1][i] !=9:
                if parHaps[0][i] == parHaps[1][i]:
                    #Nothing to do if the parent is homozygous.
                    pass                    
                else:
                    #Hap 0 is the right fit.
                    if parHaps[0][i] == haps[parIndex][i] and parHaps[1][i] != haps[parIndex][i]:
                        hap0Val = ei
                        hap1Val = e
                    #hap 1 is the right fit.
                    if parHaps[1][i] == haps[parIndex][i] and parHaps[0][i] != haps[parIndex][i]:
                        hap0Val = e
                        hap1Val = ei
                    #PaternalHaplotype
                    if parIndex == 0:
                        pointSeg[i,0] *= hap0Val # 0,0
                        pointSeg[i,1] *= hap0Val # 0,1
                        pointSeg[i,2] *= hap1Val # 1,0
                        pointSeg[i,3] *= hap1Val # 1,1
                    #MaternalHaplotype
                    if parIndex == 1:
                        pointSeg[i,0] *= hap0Val # 0,0
                        pointSeg[i,1] *= hap1Val # 0,1
                        pointSeg[i,2] *= hap0Val # 1,0
                        pointSeg[i,3] *= hap1Val # 1,1
    return pointSeg
@jit(nopython=True, locals={'e': float32, 'e2':float32, 'e1e':float32, 'e2i':float32})
def loopyPeelingForwardBack(pointSeg, rate):
    #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
    e = rate
    e2 = e**2
    e1e = e*(1-e)
    e2i = 1.0 - e2

    nLoci = pointSeg.shape[0] 

    seg = np.full(pointSeg.shape, .25, dtype = np.float32)
    for i in range(nLoci):
        for j in range(4):
            seg[i,j] = pointSeg[i,j]

    tmp = np.full((4), 0, dtype = np.float32)
    new = np.full((4), 0, dtype = np.float32)

    prev = np.full((4), .25, dtype = np.float32)
    for i in range(1, nLoci):
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[i-1,j]
        
        sum_j = 0
        for j in range(4):
            sum_j += tmp[j]
        for j in range(4):
            tmp[j] = tmp[j]/sum_j

        # !                  fm  fm  fm  fm 
        # !segregationOrder: pp, pm, mp, mm

        new[0] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        # tmp = tmp/np.sum(tmp)
        # new = e2i*tmp + e2 + e1e*(tmp[0] + tmp[3])*same + e1e*(tmp[1] + tmp[2])*diff       

        for j in range(4):
            seg[i,j] *= new[j]
        # seg[:,i] *= new
        prev = new

    prev = np.full((4), .25, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[i+1,j]
        
        sum_j = 0
        for j in range(4):
            sum_j += tmp[j]
        for j in range(4):
            tmp[j] = tmp[j]/sum_j

        new[0] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[i,j] *= new[j]
        prev = new

    for i in range(nLoci):
        sum_j = 0
        for j in range(4):
            sum_j += seg[i,j]
        for j in range(4):
            seg[i,j] = seg[i,j]/sum_j

    return(seg)





    #####################################
    #####################################
    ####                            #####
    ####     PeelDown Imputation    #####
    ####                            #####
    #####################################
    #####################################



def imputeFromChildren(ind, cutoff = 0.9999) :

    nLoci = len(ind.haplotypes[0])
    ones = np.full((nLoci, 2), 0, np.int64)
    counts = np.full((nLoci, 2), 0, np.int64)

    matLib = np.transpose(np.array(ind.haplotypes))

    if len(ind.offspring) > 0:
        for child in ind.offspring:
            childHap = None
            if child.sire is ind:
                childHap = child.haplotypes[0]
            if child.dam is ind:
                childHap = child.haplotypes[1]
            if childHap is None:
                print("Something bad happened. Child of parent does not have parent as a parent")

            pointEstimates = getPointSegs(childHap, matLib, error = 0.01)

            rate = 1.0/nLoci

            probs = haploidForwardBack(pointEstimates, rate)
            calledHap = callReverseProbs(probs, childHap, cutoff = cutoff)

            addToCountsIfNotMissing_2D(calledHap, ones, counts)
        
        for e in range(2):
            calledHap = callCounts(ones[:,e], counts[:,e], threshold = .95)
            Imputation.filInIfMissing(orig = ind.haplotypes[e], new = calledHap)

@jit(nopython=True)
def callCounts(ones, counts, threshold):
    nLoci = len(ones)
    calledHap = np.full(nLoci, 9, np.int8)
    for i in range(nLoci):
        if ones[i] > counts[i]*threshold:
            calledHap[i] = 1
        if ones[i] < counts[i]*(1-threshold):
            calledHap[i] = 0
    return calledHap

@jit(nopython=True)
def callReverseProbs(probs, refHap, cutoff):
    nLoci, nHaps = probs.shape
    hap = np.full(probs.shape, 9, dtype = np.int8)
    #Work on calling first, and then use that. Don't use dosages.
    for i in range(nLoci):
        for j in range(nHaps):
            score = probs[i, j]
            if score > cutoff:
                hap[i, j] = refHap[i]
    return hap

@jit(nopython=True)
def addToCountsIfNotMissing_2D(haplotypes, ones, counts):
    nLoci, nHaps = haplotypes.shape
    for i in range(nLoci):
        for j in range(nHaps):
            if haplotypes[i, j] != 9:
                counts[i,j] += 1
                ones[i,j] += haplotypes[i,j]


    #####################################
    #####################################
    ####                            #####
    ####  HaploidFamily Imputation  #####
    ####                            #####
    #####################################
    #####################################

def getAncestors(ind, depth):
    if ind is None:
        return []
    if depth == 0:
        return [ind]
    else:
        ancestors = []
        if ind.sire is not None:
            ancestors += getAncestors(ind.sire, depth-1)
        if ind.dam is not None:
            ancestors += getAncestors(ind.dam, depth-1)
        return ancestors

def imputeFromAncestors(ind, depth = 1, cutoff = 0.9999):

    #sire
    if ind.sire is not None:
        haploidFamilyHMM(ind.haplotypes[0], getAncestors(ind.sire, depth=depth), cutoff = cutoff)
        Imputation.ind_align(ind)
    if ind.dam is not None:
        haploidFamilyHMM(ind.haplotypes[1], getAncestors(ind.dam, depth=depth), cutoff = cutoff)
        Imputation.ind_align(ind)


def haploidFamilyHMM(hap, individuals, cutoff = 0.9):

    hapLib = []
    for ind in individuals:
        for e in range(2):
            # if np.mean(ind.haplotypes[e] == 9) <.1:
            hapLib.append(ind.haplotypes[e])    

    if len(hapLib) >0 :
        matLib = np.transpose(np.array(hapLib))
        

        #Maybe I'm being silly here...
        nLoci, nHaps = matLib.shape

        pointEstimates = getPointSegs(hap, matLib, error = 0.01)

        rate = 1.0/nLoci

        probs = haploidForwardBack(pointEstimates, rate)
        # print("collapsed")
        # print(probs[0:30, :])

        calledHap = callProbs(probs, matLib, cutoff = cutoff)

        Imputation.filInIfMissing(hap, calledHap)

# @jit(nopython=True)
# def callProbs_dosages(probs, matLib, cutoff):
#     nLoci, nHaps = matLib.shape
#     hap = np.full(nLoci, 9, dtype = np.int8)
#     #Work on calling first, and then use that. Don't use dosages.
#     for i in range(nLoci):
#         score = 0
#         weight = 0
#         for j in range(nHaps):
#             val = matLib[i, j]
#             if val != 9:
#                 score += matLib[i, j]*probs[i,j]
#                 weight += probs[i,j]
#         if weight > 0:
#             score = score/weight
#             if score > cutoff:
#                 hap[i] = 1
#             if score <= 1-cutoff:
#                 hap[i] = 0
#     return hap
@jit(nopython=True)
def callProbs(probs, matLib, cutoff):
    nLoci, nHaps = matLib.shape
    hap = np.full(nLoci, 9, dtype = np.int8)
    #Work on calling first, and then use that. Don't use dosages.
    for i in range(nLoci):
        for j in range(nHaps):
            score = probs[i, j]
            if score > cutoff:
                hap[i] = matLib[i, j]
    return hap
@jit(nopython=True)
def getPointSegs(hap, haps, error):
    nLoci, nHaps = haps.shape
    pointEstimates = np.full(haps.shape, .5, dtype = np.float32)
    for i in range(nLoci):
        if hap[i] != 9:
            include = True
            for j in range(nHaps):
                if haps[i, j] == 9:
                    include = False
                    break
            if include:
                for j in range(nHaps):
                    if haps[i, j] == 9:
                        pointEstimates[i, j] = .5
                    if hap[i] == haps[i, j]:
                        pointEstimates[i, j] = 1 - error
                    else:
                        pointEstimates[i, j] = error
    return pointEstimates

@jit(nopython=True, locals={'e': float32, 'e1':float32})
def haploidForwardBack(pointSeg, rate):
    #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
    e = rate
    e1 = 1-e
    nLoci = pointSeg.shape[0] 
    nHaps = pointSeg.shape[1]

    seg = np.full(pointSeg.shape, .25, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nHaps):
            seg[i,j] = pointSeg[i,j]

    tmp = np.full(nHaps, 0, dtype = np.float32)
    new = np.full(nHaps, 0, dtype = np.float32)

    prev = np.full(nHaps, .25, dtype = np.float32)
    
    for i in range(1, nLoci):
        for j in range(nHaps):
            tmp[j] = prev[j]*pointSeg[i-1,j]
        
        sum_j = 0
        for j in range(nHaps):
            sum_j += tmp[j]
        for j in range(nHaps):
            tmp[j] = tmp[j]/sum_j

        # !                  fm  fm  fm  fm 
        # !segregationOrder: pp, pm, mp, mm

        for j in range(nHaps):
            new[j] = e + e1*tmp[j]

        # tmp = tmp/np.sum(tmp)
        # new = e2i*tmp + e2 + e1e*(tmp[0] + tmp[3])*same + e1e*(tmp[1] + tmp[2])*diff       

        for j in range(nHaps):
            seg[i,j] *= new[j]
        # seg[:,i] *= new
        prev = new

    prev = np.full((nHaps), .25, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        for j in range(nHaps):
            tmp[j] = prev[j]*pointSeg[i+1,j]
        
        sum_j = 0
        for j in range(nHaps):
            sum_j += tmp[j]
        for j in range(nHaps):
            tmp[j] = tmp[j]/sum_j
        
        for j in range(nHaps):
            new[j] = e + e1*tmp[j]

        for j in range(nHaps):
            seg[i,j] *= new[j]
        prev = new

    for i in range(nLoci):
        sum_j = 0
        for j in range(nHaps):
            sum_j += seg[i,j]
        for j in range(nHaps):
            seg[i,j] = seg[i,j]/sum_j

    return(seg)

