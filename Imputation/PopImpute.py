import numba
from numba import jit, int8, int32, boolean, jitclass
import numpy as np

from tinyhouse import HaplotypeLibrary
from tinyhouse import BurrowsWheelerLibrary
from Imputation import Imputation

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

    regions = getMissingRegions(indHap)
    newHap = np.full(nLoci, 9, dtype = np.int8)
    
    expTable = getExpTable(101)

    if len(regions) > 0:
        errorLibrary = HaplotypeLibrary.ErrorLibrary(indHap, bwLib.library.haps)
        windowErrors = errorLibrary.getWindowValue(50)

        for region in regions:

            start, stop = region

            vals = windowErrors[:,start]
            weights = silly(vals, expTable)
            # weights = error**vals

            bestHap = bwLib.getBestHaplotype(weights, start, stop)
            newHap[start:stop] = bestHap

    return newHap

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

# hap = np.array([0, 1, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 0, 0, 1])
# getMissingRegions(hap)

