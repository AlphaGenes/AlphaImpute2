import numba
from numba import jit, int8, int32, boolean, jitclass
import numpy as np

from ..tinyhouse import HaplotypeLibrary
from ..tinyhouse import BurrowsWheelerLibrary
from ..tinyhouse import InputOutput

from . import Imputation
import random
import collections

import concurrent.futures
from itertools import repeat

np.seterr(all='raise')

try:
    profile
except:
    def profile(x): 
        return x


@profile
def phaseHD(phasingInfo, nHet, setHap = False) :
    print(nHet, len(phasingInfo.referenceHaplotypes))
    hapLib = HaplotypeLibrary.HaplotypeLibrary()
    

    for refHap in phasingInfo.referenceHaplotypes:
        hapLib.append(refHap)
    
    if not phasingInfo.useReferenceOnly:
        for idn, hapPair in phasingInfo.currentHaplotypes.items():
            for e in (0, 1) :
                hapLib.append(hapPair[e].copy())
    
    hapLib.removeMissingValues()

    bwLibrary = BurrowsWheelerLibrary.BurrowsWheelerLibrary(hapLib.library)
    
    parallel = InputOutput.args.maxthreads > 1
    
    print(len(phasingInfo.currentHaplotypes))
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=InputOutput.args.maxthreads) as executor:
            idns = [idn for idn in phasingInfo.currentHaplotypes]
            results = executor.map(phaseIdn, idns, repeat(phasingInfo), repeat(bwLibrary), repeat(nHet), repeat(setHap))

    else:
        for idn in phasingInfo.currentHaplotypes:
            phaseIdn(idn, phasingInfo, bwLibrary, nHet, setHap)

def phaseIdn(idn, phasingInfo, bwLibrary, nHet, setHap):
    geno = phasingInfo.geno[idn]
    hap0, hap1 = phasingInfo.baseHaplotypes[idn]
    cores, markerList = decomposeRegions_justHets(geno, nHet, maxRegion = int(.10*len(geno)))
    #This is hacky. For a lot of reasons, cases where this happen are... odd.
    if len(cores) == 1:
        return

    graph = HaplotypeGraph(cores, markerList, hap0, hap1)
    graph.appendBWLibrary(bwLibrary)
    # graph.checkAppend(hapLib.library, bwLibrary)

    if setHap:
        hap = graph.veterbi()
        currentHaplotypes = [hap0.copy(), hap1.copy()]
        Imputation.filInIfMissing(currentHaplotypes[0], hap)
        Imputation.fillInCompPhase(currentHaplotypes[1], geno, currentHaplotypes[0])
        phasingInfo.currentHaplotypes[idn] = currentHaplotypes
    else:
        #This is more convoluted than I'd like.
        hap = graph.samplePhase()
        currentHaplotypes = [hap0.copy(), hap1.copy()]
        Imputation.filInIfMissing(currentHaplotypes[0], hap)
        Imputation.fillInCompPhase(currentHaplotypes[1], geno, currentHaplotypes[0])
        phasingInfo.currentHaplotypes[idn] = currentHaplotypes



@jit(nopython=True)
def getHets(geno):
    nLoci = len(geno)
    isHet = np.full(nLoci, 0, dtype = np.int8)
    for i in range(nLoci):
        if geno[i] == 1:
            isHet[i] = 1
    return isHet

@profile
def decomposeRegions(geno, nHets, maxRegion = 1000) :
    nLoci = len(geno)
    isHet = getHets(geno)

    markers = np.full(nHets, 0)
    pos = 0
    start = 0
    markerList = []
    cores = []
    for i in range(nLoci):
        if isHet[i]:
            markers[pos] = i
            pos += 1
        if pos == nHets or i-start > maxRegion or i == nLoci-1:
            cores.append((start,i))
            if pos != nHets:
                markersNew = np.full(pos, 0)
                markersNew[:] = markers[0:pos]
                markers = markersNew
            markerList.append(markers)
            start = i + 1
            pos = 0
            markers = np.full(nHets, 0)

    return (cores, markerList)


@jit(nopython=True)
def getHetList(geno, nHets, maxRegion):
    nLoci = len(geno)
    hetLoci = np.full(nLoci, 0, dtype = np.int64)
    nHetLoci = 0

    cores = np.full((nLoci, nHets), -1, dtype = np.int64)
    coreSizes = np.full(nLoci, 0, dtype = np.int64)

    nCores = 0
    coreStart = 0
    corePos = 0

    for i in range(nLoci):
        if geno[i] == 1:
            hetLoci[nHetLoci] = i
            nHetLoci += 1
    
    for i in range(nHetLoci):
        hetPos = hetLoci[i]
        if (corePos > 0) and (hetPos - coreStart > maxRegion):
            corePos = 0
            nCores += 1

        cores[nCores, corePos] = hetPos
        coreSizes[nCores] += 1
        if corePos == 0:
            coreStart = hetPos
        corePos += 1
        if corePos == nHets:
            corePos = 0
            nCores += 1
    if corePos != 0:
        #If we didn't stop at a clean break add the extra partial core to the count.
        nCores +=1

    return (cores, coreSizes, nCores, nHetLoci) #Last core may only be partially filled; with max regions, earlier cores may also only be partially filled.


@profile
def decomposeRegions_justHets(geno, nHets, maxRegion = 1000) :
    nLoci = len(geno)
    hetList, hetListSizes, nCores, nHetLoci = getHetList(geno, nHets, maxRegion)
    #Need to rename stuff.
    markerList = [hetList[i, 0:hetListSizes[i]] for i in range(nCores)]
    # print(nCores)
    # print(markerList)
    if nCores == 1 or nHetLoci == 0:
        cores = [(0,nLoci)]
    else:
        #Cores start at first het loci
        # cores = [(0, markerList[1][0])]
        # midCores = [(markerList[i][0], markerList[i+1][0]) for i in range(1, nCores-1)]
        # cores.extend(midCores)
        # cores.append((markerList[-1][0], nLoci))
        
        #Cores end at last het loci.
        cores = [(0, markerList[0][-1]+1)]
        midCores = [(markerList[i-1][-1]+1, markerList[i][-1]+1) for i in range(1, nCores-1)]
        cores.extend(midCores)
        cores.append((markerList[-2][-1]+1, nLoci))

    return (cores, markerList)


# geno = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], dtype = np.int8)

# decomposeRegions(geno, 3, maxRegion = 1000)
# decomposeRegions_justHets(geno, 3, maxRegion = 1000)


class HaplotypeGraph(object):
    def __init__(self, cores, markersPerCore, hap1, hap2) :
        self.cores = cores
        self.markersPerCore = markersPerCore
        self.nCores = len(cores)

        self.hapsPerCore = [HaplotypeLibrary.HaplotypeDict() for i in range(self.nCores)]
        self.transitions = [dict() for i in range(self.nCores-1)]

        self.hap1 = hap1
        self.hap2 = hap2

        self.nLoci = len(hap1)
        #Unconvinced, but probably healthy.
        # self.append(hap1)
        # self.append(hap2)
    
    def checkAppend(self, hapLib, bwLibrary):
        for i in range(self.nCores-1):
            start = self.cores[i][0]
            stop = self.cores[i+1][1]


            hapLibHaps0 = collections.Counter()
            hapLibHaps1 = collections.Counter()

            for hap in hapLib:
                align = alignToHaps(hap[start:stop], self.hap1[start:stop], self.hap2[start:stop])            
                if align == 0 or align == 2:
                    hapLibHaps0.update([tuple(hap[start:stop])])

                if align == 1 or align == 2:
                    hapLibHaps1.update([tuple(hap[start:stop])])

            bwLibHaps0 = collections.Counter()
            hapPlusCounts = bwLibrary.getHaplotypeMatches(self.hap1, start, stop)
            for pair in hapPlusCounts :
                for count in range(pair[1]):
                    bwLibHaps0.update([tuple(pair[0][start:stop])])
        
            bwLibHaps1 = collections.Counter()
            hapPlusCounts = bwLibrary.getHaplotypeMatches(self.hap2, start, stop)
            for pair in hapPlusCounts :
                for count in range(pair[1]):
                    bwLibHaps1.update([tuple(pair[0][start:stop])])

            if not hapLibHaps0 ==bwLibHaps0:
                print(start, stop, len(self.hap1))
                print(i, len(hapLibHaps0), len(hapLibHaps1), len(bwLibHaps0), len(bwLibHaps1))
                print(i, self.nCores-1, hapLibHaps0 ==bwLibHaps0)
                print(i, self.nCores-1, hapLibHaps1 ==bwLibHaps1)
                
                print(self.hap1[start:stop])
                print(self.hap2[start:stop])
                print(" ")

                for key, value in hapLibHaps0.most_common():
                    print(key, value)
                print(" ")
                for key, value in bwLibHaps0.most_common():
                    print(key, value)
                # print(hapLibHaps0)
                # print(bwLibHaps0)
                raise Exception
    
    def append(self, hap):
        for i in range(self.nCores-1) :    
            start = self.cores[i][0]
            stop = self.cores[i+1][1]

            align = alignToHaps(hap[start:stop], self.hap1[start:stop], self.hap2[start:stop])            

            if align == 0 or align == 2:
                hapCore1 = hap[self.markersPerCore[i]]
                hapCore2 = hap[self.markersPerCore[i+1]]
                self.addToCoresAndTransitions(hapCore1, hapCore2, i, count = 1)
            if align == 1 or align == 2:
                hapCore1 = 1-hap[self.markersPerCore[i]]
                hapCore2 = 1-hap[self.markersPerCore[i+1]]
                self.addToCoresAndTransitions(hapCore1, hapCore2, i, count = 1)
    @profile
    def appendBWLibrary(self, bwLibrary):
        for i in range(self.nCores-1) :    
            #Not inclusive.
            start = self.cores[i][0]
            stop = self.cores[i+1][1]

            if numMissing(self.hap1, self.markersPerCore[i]) + numMissing(self.hap1, self.markersPerCore[i+1]) == 0:
                selfCount = 1000
                onlySelf = True
            else:
                selfCount = 1
                onlySelf = False
            
            #Add self
            hapCore1 = self.hap1[self.markersPerCore[i]]
            hapCore2 = self.hap1[self.markersPerCore[i+1]]
            self.addToCoresAndTransitions(hapCore1, hapCore2, i, count = selfCount)

            if not onlySelf:
                # print("Imputing!", numMissing(self.hap1, self.markersPerCore[i]), numMissing(self.hap1, self.markersPerCore[i+1]))
                #e = 0
                hapsPlusCounts = bwLibrary.getHaplotypeMatches(self.hap1, start, stop)
                for pair in hapsPlusCounts:
                    hap, count = pair
                    hapCore1 = hap[self.markersPerCore[i]]
                    hapCore2 = hap[self.markersPerCore[i+1]]
                    self.addToCoresAndTransitions(hapCore1, hapCore2, i, count = count)

                #e = 1
                hapsPlusCounts = bwLibrary.getHaplotypeMatches(self.hap2, start, stop)
                for pair in hapsPlusCounts:
                    hap, count = pair
                    hapCore1 = 1-hap[self.markersPerCore[i]]
                    hapCore2 = 1-hap[self.markersPerCore[i+1]]
                    self.addToCoresAndTransitions(hapCore1, hapCore2, i, count = count)

    @profile
    def addToCoresAndTransitions(self, hapCore1, hapCore2, i, count = 1):

        hap1Index = self.hapsPerCore[i].append(hapCore1)
        hap2Index = self.hapsPerCore[i+1].append(hapCore2)

        combined = (hap1Index, hap2Index)
        if combined in self.transitions[i]:
            self.transitions[i][combined] += count
        else:
            self.transitions[i][combined] = count

    def nextRandom(self):
        return(random.randrange(2))
    def toZeroOneTuple(self, hap) :
        for i in range(len(hap)):
            if hap[i] != 0 and hap[i] != 1:
                hap[i] = self.nextRandom()
        return(tuple(hap))
    def getTransitionMat(self, i):
        transitionMat = np.full((self.hapsPerCore[i].nHaps, self.hapsPerCore[i+1].nHaps), .1, dtype = np.float32)
        for key, value in self.transitions[i].items():
            c1, c2 = key
            transitionMat[c1, c2] = value
        return transitionMat
    @profile
    def getForwardProbabilities(self):
        probs = [np.full(core.nHaps, 0) for core in self.hapsPerCore]
        probs[0][:] = 1
        for i in range(self.nCores-1):
            transitionMat = self.getTransitionMat(i)
            transitionMat = transitionMat/np.sum(transitionMat, 1)[:,None]
            probs[i+1] = np.dot(probs[i], transitionMat)
            probs[i+1] /= np.sum(probs[i+1])
            # vect_norm(probs[i+1])
        return(probs)
    @profile
    def getBackwardProbabilities(self):
        probs = [np.full(core.nHaps, 0) for core in self.hapsPerCore]
        probs[self.nCores-1][:] = 1
        for i in range(self.nCores-2, -1, -1):
            transitionMat = self.getTransitionMat(i)
            transitionMat = transitionMat/np.sum(transitionMat, 0)[None,:]
            probs[i] = np.dot(transitionMat, probs[i+1])
            probs[i] /= np.sum(probs[i])
            # vect_norm(probs[i])

        return(probs)

    def callPhase(self, thresh):
        fwd = self.getForwardProbabilities() 
        bwd = self.getBackwardProbabilities()
        calls = []

        #Convert to probabilities.
        for cIndex in range(len(self.hapsPerCore)):
            # probs = fwd[cIndex]*bwd[cIndex]
            probs = fwd[cIndex]
            probs = probs/np.sum(probs)
            phase = np.full(len(self.markersPerCore[cIndex]),0, dtype = np.float32)
            for i in range(self.hapsPerCore[cIndex].nHaps) :
                core = self.hapsPerCore[cIndex].get(i)
                phase += np.array(core)*probs[i]
            for i in range(len(phase)):
                # print(phase[i])
                if phase[i] <= (1-thresh) :
                    calls.append((self.markersPerCore[cIndex][i], 0))
                if phase[i] >= thresh :
                    calls.append((self.markersPerCore[cIndex][i], 1))
        return calls

    @profile
    def samplePhase(self):
        fwd = self.getForwardProbabilities() 


        bwd = [np.full(core.nHaps, 0) for core in self.hapsPerCore]
        bwd[self.nCores-1][:] = 1
        hapIds = [-1 for core in self.hapsPerCore]
        for i in range(self.nCores-1, -1, -1):

            #sample a value
            localProbs = fwd[i]*bwd[i]
            localProbs /= np.sum(localProbs)

            val = np.random.choice(len(localProbs), p = localProbs)
            hapIds[i] = val
            bwd[i][:] = 0
            bwd[i][val] = 1

            if i >0:
                transitionMat = self.getTransitionMat(i-1)
                transitionMat = transitionMat/np.sum(transitionMat, 0)[None,:]
                bwd[i-1] = np.dot(transitionMat, bwd[i])
                # vect_norm(bwd[i-1])
                bwd[i-1] /= np.sum(bwd[i-1])

        hap = self.recreateHap(hapIds)

        return hap
    def veterbi(self):
        fwd = self.getForwardProbabilities() 


        bwd = [np.full(core.nHaps, 0) for core in self.hapsPerCore]
        bwd[self.nCores-1][:] = 1
        hapIds = [-1 for core in self.hapsPerCore]
        for i in range(self.nCores-1, -1, -1):

            #sample a value
            localProbs = fwd[i]*bwd[i]
            localProbs /= np.sum(localProbs)
            val = np.argmax(localProbs)
            hapIds[i] = val
            bwd[i][:] = 0
            bwd[i][val] = 1

            if i >0:
                transitionMat = self.getTransitionMat(i-1)
                transitionMat = transitionMat/np.sum(transitionMat, 0)[None,:]
                bwd[i-1] = np.dot(transitionMat, bwd[i])
                # vect_norm(bwd[i-1])
                bwd[i-1] /= np.sum(bwd[i-1])

        hap = self.recreateHap(hapIds)

        return hap

    def recreateHap(self, hapIds):
        hap = np.full(self.nLoci, 9, dtype = np.int8)
        for i in range(self.nCores):
            indexes = self.markersPerCore[i]
            coreHap = self.hapsPerCore[i].get(hapIds[i])
            for coreIndex, hapIndex in enumerate(indexes):
                hap[hapIndex] = coreHap[coreIndex]

        return hap

@jit(nopython=True)
def vect_norm(vect):
    val = 0
    for i in range(len(vect)):
        val += vect[i]
    for i in range(len(vect)):
        vect[i]/=val

@jit(nopython=True)
def alignToHaps(hap, hap0, hap1) :
    #Hap 1 and hap2 are the individuals's two haplotypes
    #Hap is the haplotype currently being examined.

    nLoci = len(hap)

    hap0Error = 0
    hap1Error = 0

    for i in range(nLoci) :

        if hap[i] == 9: 
            pass
        else:
            if(hap0[i] != 9) and (hap0[i] != hap[i]):
                hap0Error += 1
            if(hap1[i] != 9) and (hap1[i] != hap[i]):
                hap1Error += 1
            if hap0Error > 0 and hap1Error > 0 : return -1

    if hap0Error == 0 and hap1Error > 0:
        return 0
    if hap0Error > 0 and hap1Error == 0:
        return 1
    if hap0Error == 0 and hap1Error == 0:
        return 2
    return -1

def numMissing(hap, markers):
    val = 0
    for marker in markers:
        if hap[marker] == 9:
            val += 1
    return val
    
class PhasingInformation(object):
    def __init__(self, useReferenceOnly = False):
        self.useReferenceOnly = useReferenceOnly
        self.individuals = []

        self.baseHaplotypes = dict()
        self.currentHaplotypes = dict()
        self.referenceHaplotypes = []
        self.geno = dict()
        self.markers = []

    def addIndividuals(self, indList, markers):
        self.markers = markers
        for ind in indList:
            self.individuals.append(ind)

            self.baseHaplotypes[ind.idn] = (ind.haplotypes[0][markers].copy(), ind.haplotypes[1][markers].copy())
            self.currentHaplotypes[ind.idn] = (ind.haplotypes[0][markers].copy(), ind.haplotypes[1][markers].copy())
            self.geno[ind.idn] = ind.genotypes[markers]
    
    def addReference(self, indList):
        for ind in indList:
            self.checkAndAddReferenceHaplotype(ind.haplotypes[0][self.markers].copy())
            self.checkAndAddReferenceHaplotype(ind.haplotypes[1][self.markers].copy())

    def checkAndAddReferenceHaplotype(self, refHap):
        if np.mean(refHap == 9) < .01:
            self.referenceHaplotypes.append(refHap)

    def alignIndividuals(self):
        for ind in self.individuals:
            if ind.idn in self.currentHaplotypes:
                addWithMarkerList(ind.haplotypes[0], self.currentHaplotypes[ind.idn][0], self.markers)
                addWithMarkerList(ind.haplotypes[1], self.currentHaplotypes[ind.idn][1], self.markers)



def addWithMarkerList(baseHap, newHap, markers):
    for i in range(len(markers)):
        index = markers[i]
        if baseHap[index] == 9:
            #New Hap is in marker coding.
            baseHap[index] = newHap[i]








