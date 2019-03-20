import numba
from numba import jit, int8, int32, boolean, jitclass
import numpy as np

import math

from ..tinyhouse import HaplotypeLibrary
from ..tinyhouse import BurrowsWheelerLibrary
from . import Imputation

try:
    profile
except:
    def profile(x): 
        return x


@profile
def impute(indList, referencePanel) :
    #Grab high density haplotypes. Fill in missing values.
    hapLib = HaplotypeLibrary.HaplotypeLibrary()
    for ind in referencePanel:
        for e in (0, 1) :
            if ind.initHD or ( np.mean(ind.haplotypes[e] == 9) <.1 ):
                hapLib.append(ind.haplotypes[e].copy())
    hapLib.removeMissingValues()
    bwLib = BurrowsWheelerLibrary.BurrowsWheelerLibrary(hapLib.library)

    for ind in indList:
        for e in range(2):
            imputeInd(ind, e, hapLib, bwLib)
        Imputation.ind_align(ind)
@profile
def imputeInd(ind, e, hapLib, bwLib):
    calledHap = popImpute(ind.haplotypes[e], hapLib, bwLib)
    Imputation.filInIfMissing(ind.haplotypes[e], calledHap)

@profile
def popImpute(indHap, hapLib, bwLib) :
    nHaps = hapLib.nHaps
    nLoci = len(indHap)

    # regions = getMissingRegions(indHap)
    # regions = getSequentialRegions(nLoci, 25, 12)
    regions = getIndividualRegion(indHap, 100, 10)
    newHap = np.full(nLoci, 9, dtype = np.int8)
        # expTable = getExpTable(101)

    if len(regions) > 0 and np.sum(indHap !=9) > 0 and np.sum(indHap == 9) > 0:

        for region in regions:
            start, coreStart, coreStop, stop = region
            haps = bwLib.getHaplotypeMatches(indHap, start, stop)
            if len(haps) > 0 :
                bestHapIndex = np.argmax([hap[1] for hap in haps])
                bestHap = haps[bestHapIndex][0] 
                newHap[coreStart:coreStop] = bestHap[coreStart:coreStop]

    return newHap
# @profile
# def popImpute(indHap, hapLib, bwLib) :
#     nHaps = hapLib.nHaps
#     nLoci = len(indHap)

#     regions = getMissingRegions(indHap)
#     newHap = np.full(nLoci, 9, dtype = np.int8)
    
#     # expTable = getExpTable(101)

#     if len(regions) > 0:

#         for region in regions:
#             start, stop = region

#             start = max(start -25, 0)
#             stop = min(stop + 25, nLoci-1)

#             haps = bwLib.getHaplotypeMatches(indHap, start, stop)
#             # print(haps)
#             if len(haps) > 0 :

#                 bestHapIndex = np.argmax([hap[1] for hap in haps])
#                 # print(len(haps[bestHapIndex][0]))
#                 bestHap = haps[bestHapIndex][0][start:stop]

#                 newHap[start:stop] = bestHap

#     return newHap

@jit(nopython=True)
def silly(vect, table):
    out = np.full(len(vect), 0, dtype = np.float32)
    for i in range(len(vect)):
        out[i] = table[vect[i]]
    return out

@jit(nopython=True)
def getExpTable(maxVal):
    error = 0.01
    out = np.full(maxVal, 0, dtype = np.float32)
    for i in range(maxVal):
        out[i] = error**i
        if out[i] < 0.00000000000001:
            break
    return out

# @jit(nopython=True)
def getSequentialRegions(nLoci, coreSize, tailSize):
    regions = []
    totalSize = coreSize + tailSize
    nRegions = math.ceil(nLoci/coreSize)
    for i in range(nRegions):
        tailStart = max(0, i*coreSize - tailSize)
        coreStart = max(0, i*coreSize)
        coreEnd = min(nLoci, (i+1)*coreSize)
        tailEnd = min(nLoci, (i+1)*coreSize + tailSize)
        regions.append([tailStart, coreStart, coreEnd, tailEnd ])
    return(regions)

def getIndCore(hap, coreStart, coreStop, targetNMarkers):
    nLoci = len(hap)

    numberNonMissingMarkers = 0
    for i in range(coreStart, coreStop) :
        if hap[i] != 9:
            numberNonMissingMarkers += 1

    regionStart = coreStart
    regionStop = coreStop
    # print(numberNonMissingMarkers, targetNMarkers)
    # print(regionStart, regionStop, coreStart, coreStop)

    #Allow for triple the region size.
    while (numberNonMissingMarkers < targetNMarkers) & ((regionStop - regionStart) < 3*(coreStop - coreStart))  & (regionStart != 0 | regionStop != nLoci):
        # print(numberNonMissingMarkers, targetNMarkers)
        # print(regionStart, regionStop, coreStart, coreStop)
        if regionStart > 0:
            regionStart -= 1
            if hap[regionStart] != 9:
                numberNonMissingMarkers += 1

        if regionStop < nLoci:
            regionStop += 1
            if hap[regionStop -1] != 9:
                numberNonMissingMarkers += 1

    return regionStart, regionStop

def getIndividualRegion(hap, coreSize, targetNMarkers):
    #Core stop is non-inclusive.
    nLoci = len(hap)
    regions = []
    totalSize = coreSize
    nRegions = math.ceil(nLoci/coreSize)
    for i in range(nRegions):
        coreStart = max(0, i*coreSize)
        coreStop = min(nLoci, (i+1)*coreSize)
        tailStart, tailStop = getIndCore(hap, coreStart, coreStop, targetNMarkers)
        regions.append([tailStart, coreStart, coreStop, tailStop ])
    return(regions)

# getSequentialRegions(1000, 50, 50)

@jit(nopython=True)
def getMissingRegions(hap):
    regions = []

    currentlyTracking = False
    start = -1
    
    for i in range(len(hap)):
        if currentlyTracking:
            if hap[i] != 9:
                #Non inclusive, so this is the first loci
                regions.append((start, i))
                currentlyTracking = False
                start = -1
        else:
            if hap[i] == 9:
                start = i
                currentlyTracking = True
    return regions
