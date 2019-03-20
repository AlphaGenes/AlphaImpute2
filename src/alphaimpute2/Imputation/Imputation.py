import numba
from numba import jit, int8, int32, boolean, jitclass, float32, int64
import numpy as np


def ind_fillInGenotypesFromPhase(ind):
    fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])

def ind_align(ind):
    #Note: We never directly set genotypes so no need to go from genotypes -> phase
    fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])
    fillInCompPhase(ind.haplotypes[0], ind.genotypes, ind.haplotypes[1])
    fillInCompPhase(ind.haplotypes[1], ind.genotypes, ind.haplotypes[0])

@jit(nopython=True)
def filInIfMissing(orig, new):
    for i in range(len(orig)):
        if orig[i] == 9:
            orig[i] = new[i]

@jit(nopython=True)
def fillInGenotypesFromPhase(geno, phase1, phase2):
    for i in range(len(geno)):
        if geno[i] == 9:
            if phase1[i] != 9 and phase2[i] != 9:
                geno[i] = phase1[i] + phase2[i]
@jit(nopython=True)
def fillInCompPhase(target, geno, compPhase):
    for i in range(len(geno)):
        if target[i] == 9:
            if geno[i] != 9:
                if compPhase[i] != 9:
                    target[i] = geno[i] - compPhase[i]
    
@jit(nopython=True)
def fillInPhaseFromGenotypes(phase, geno):
    for i in range(len(geno)):
        if phase[i] == 9 :
            if geno[i] == 0: phase[i] = 0
            if geno[i] == 2: phase[i] = 1



def ind_resetPhasing(ind):
    ind.haplotypes[0][:] = 9
    ind.haplotypes[1][:] = 9
    fillInPhaseFromGenotypes(ind.haplotypes[0], ind.genotypes)
    fillInPhaseFromGenotypes(ind.haplotypes[1], ind.genotypes)
    ind_randomlyPhaseRandomPoint(ind)

def ind_randomlyPhaseRandomPoint(ind):
    maxLength = len(ind.genotypes)
    midpoint = np.random.normal(maxLength/2, maxLength/10)
    while midpoint < 0 or midpoint > maxLength:
        midpoint = np.random.normal(maxLength/2, maxLength/10)
    midpoint = int(midpoint)
    randomlyPhaseMidpoint(ind.genotypes, ind.haplotypes, midpoint)

def ind_randomlyPhaseMidpoint(ind, midpoint= None):
    if midpoint is None: midpoint = int(len(ind.genotypes)/2)
    randomlyPhaseMidpoint(ind.genotypes, ind.haplotypes, midpoint)

@jit(nopython=True)
def randomlyPhaseMidpoint(geno, phase, midpoint):

    index = 0
    e = 1
    changed = False
    while not changed :
        if geno[midpoint + index * e] == 1:
            phase[0][midpoint + index * e] = 0
            phase[1][midpoint + index * e] = 1
            changed = True
        e = -e
        if e == -1: index += 1
        if index >= midpoint: changed = True


    #####################################
    #####################################
    ####                            #####
    ####   Random Haplotype math?   #####
    ####                            #####
    #####################################
    #####################################




def ind_getDosages(ind, maf = None) :
    nLoci = len(ind.genotypes)

    if maf is None: 
        maf = np.full(nLoci, .5)
    ind.genotypeDosages = np.zeros(nLoci)
    ind.haplotypeDosages = (np.zeros(nLoci), np.zeros(nLoci))

    if ind.sire is not None:
        if ind.sire.haplotypeDosages is None:
            getDosages(ind.sire)
        sireHaps = ind.sire.haplotypeDosages
    else:
        sireHaps = (maf, maf)
    getHaplotypeDosage(ind.haplotypes[0], ind.haplotypeDosages[0], sireHaps, ind.segregation[0])

    if ind.dam is not None:
        if ind.dam.haplotypeDosages is None:
            getDosages(ind.dam)
        damHaps = ind.dam.haplotypeDosages
    else:
        damHaps = (maf, maf)
    getHaplotypeDosage(ind.haplotypes[1], ind.haplotypeDosages[0], damHaps, ind.segregation[1])

    getGenotypeDosageFromHaplotype(ind.genotypes, ind.genotypeDosages, ind.haplotypeDosages)

def ind_assignOrigin(ind):
    nLoci = len(ind.genotypes)
    ind.hapOfOrigin = (np.zeros(nLoci, dtype = np.int32), np.zeros(nLoci, dtype = np.int32))
    ind.hapOfOrigin[0][:] = int(ind.idx)*2 + 0
    ind.hapOfOrigin[1][:] = int(ind.idx)*2 + 1

    if ind.sire is not None:
        if ind.sire.hapOfOrigin is None:
            ind_assignOrigin(ind.sire)
        getHaplotypeOfOrigin(ind.hapOfOrigin[0], ind.sire.hapOfOrigin, ind.segregation[0])
    
    if ind.dam is not None:
        if ind.dam.hapOfOrigin is None:
            ind_assignOrigin(ind.dam)
        getHaplotypeOfOrigin(ind.hapOfOrigin[1], ind.dam.hapOfOrigin, ind.segregation[1])


@jit(nopython=True)
def getHaplotypeOfOrigin(hapOrigin, parOrigins, segregation) :
    for i in range(len(hapOrigin)):
        seg = segregation.getSeg(i)
        if seg == -1:
            pass
            # hapOrigin[i] = -1
        else:
            hapOrigin[i] = parOrigins[seg][i]

@jit(nopython=True)
def getSegregation(nLoci, segregation) :
    output = np.full(nLoci, -1, dtype=np.int8)
    for i in range(len(output)):
        output[i] = segregation.getSeg(i)

    return output
@jit(nopython=True)
def getHaplotypeDosage(hap, dosages, parDosages, segregation):
    for i in range(len(hap)):
        if hap[i] != 9:
            dosages[i] = hap[i]
        else:
            seg = segregation.getSegDosage(i)
            dosages[i] = parDosages[0][i]*(1-seg) + parDosages[1][i]*seg

@jit(nopython=True)
def getGenotypeDosageFromHaplotype(geno, dosages, hapDosages):
    for i in range(len(geno)) :
        if geno[i] != 9:
            dosages[i] = geno[i]
        else:
            dosages[i] = hapDosages[0][i] + hapDosages[1][i]

def pedCompareGenotypes(refPed, truePed) :
    nLoci = 0
    y = 0
    errors = 0

    for idx in refPed.individuals:
        if refPed.individuals[idx].initHD and refPed.individuals[idx].isFounder():
            if idx in truePed.individuals:
                refInd = refPed.individuals[idx]
                trueInd = truePed.individuals[idx]

                nLoci += len(refInd.genotypes)
                y += np.sum(refInd.genotypes != 9)
                errors += np.sum(np.logical_and(refInd.genotypes != 9, refInd.genotypes != trueInd.genotypes))
    return((y/nLoci, 1 - errors/y))



def pedComparePhase(refPed, truePed) :
    nLoci = 0
    y = 0
    errors = 0

    for idx in refPed.individuals:
        if refPed.individuals[idx].initHD and refPed.individuals[idx].isFounder():
            if idx in truePed.individuals:
                refInd = refPed.individuals[idx]
                trueInd = truePed.individuals[idx]

                nLoci += len(refInd.haplotypes[0])
                nLoci += len(refInd.haplotypes[1])

                y += np.sum(refInd.haplotypes[0] != 9)
                y += np.sum(refInd.haplotypes[1] != 9)

                error1 = np.sum(np.logical_and(refInd.haplotypes[0] != 9, refInd.haplotypes[0] != trueInd.haplotypes[0]))
                error1 += np.sum(np.logical_and(refInd.haplotypes[1] != 9, refInd.haplotypes[1] != trueInd.haplotypes[1]))

                error2 = np.sum(np.logical_and(refInd.haplotypes[0] != 9, refInd.haplotypes[0] != trueInd.haplotypes[1]))
                error2 += np.sum(np.logical_and(refInd.haplotypes[1] != 9, refInd.haplotypes[1] != trueInd.haplotypes[0]))

                errors += min(error1, error2)
    return((y/nLoci, 1 - errors/y))




def pedCompareDosages(refPed, truePed) :
    n = 0
    cor = 0
    for idx in refPed.individuals:
        if idx in truePed.individuals:
            refInd = refPed.individuals[idx]
            trueInd = truePed.individuals[idx]
            ind_getDosages(refInd)
            cor += np.corrcoef(refInd.genotypeDosages, trueInd.genotypes) 
            n += 1
    return(cor/n)


#Exclude is trying to prevent gratuitous backtracking.
def getRelatives(focal, maxDist, excludeInds, direction = "both"):
    val = set()

    print(focal)
    print(excludeInds)
    if direction in ("both", "up") and focal.sire is not None:
        if focal.sire not in excludeInds:
            val.add(focal.sire)
    if direction in ("both", "up") and focal.dam is not None:
        if focal.dam not in excludeInds:
            val.add(focal.dam)
    
    if direction in ("both", "down") :
        for child in focal.offspring:
            if child not in excludeInds:
                val.add(child)

    newExclude = excludeInds.union(val)
    if maxDist > 0:
        for ind in val:
            val = val.union(getRelatives(ind, maxDist-1, newExclude))
    return val






#     #####################################
#     #####################################
#     ####                            #####
#     ####     Pedigree Imputation    #####
#     ####                            #####
#     #####################################
#     #####################################




# def parentHomozygoticFillIn(ind) :
#     if ind.sire is not None:
#         ind_fillInGenotypesFromPhase(ind.sire)
#         fillInPhaseFromGenotypes(ind.haplotypes[0], ind.sire.genotypes)

#     if ind.dam is not None:
#         ind_fillInGenotypesFromPhase(ind.dam)
#         fillInPhaseFromGenotypes(ind.haplotypes[1], ind.dam.genotypes)

#     fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])


# def ind_peelDown(ind, fill = 0, reqNumMarkers = 1):

#     if ind.sire is not None:
#         ind.segregation[0] = jit_segregation()
#         # peelDown_fimpute(ind.haplotypes[0], ind.sire.haplotypes, ind.segregation[0])
#         peelDown(ind.haplotypes[0], ind.sire.haplotypes, ind.segregation[0], fill= fill, reqNumMarkers = reqNumMarkers)
#         fillInCompPhase(ind.haplotypes[1], ind.genotypes, ind.haplotypes[0]) #fill in dam phase from sire

#     if ind.dam is not None:
#         ind.segregation[1] = jit_segregation()
#         # peelDown_fimpute(ind.haplotypes[1], ind.dam.haplotypes, ind.segregation[1])
#         peelDown(ind.haplotypes[1], ind.dam.haplotypes, ind.segregation[1], fill= fill, reqNumMarkers = reqNumMarkers)
#         fillInCompPhase(ind.haplotypes[0], ind.genotypes, ind.haplotypes[1]) #fill in sire phase from dam
    
#     fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])


# @jit(nopython=True)
# def peelDown(hap, parHaps, segregation, fill = 0, reqNumMarkers = 1):
#     #Paramaters
#     # reqNumMarkers = 1
#     maxGap = 2*len(hap)
#     #Work out overhang later

#     numMarkers = 0
#     start = 0
#     end = 0
#     currentSeg = -1
    

#     firstStretch = True
#     for i in range(len(hap)):
#         seg = 9
#         #generate seg
#         if i - end > maxGap:
#             seg = -1
#         if hap[i] != 9 and (parHaps[0][i] == 9 or parHaps[1][i] == 9):
#             if currentSeg == 0 and parHaps[0][i] != 9 and hap[i] != parHaps[0][i]:
#                 seg = 1
#             if currentSeg == 1 and parHaps[1][i] != 9 and hap[i] != parHaps[1][i]:
#                 seg = 0
#         elif hap[i] != 9:
#             if hap[i] != parHaps[0][i] and hap[i] != parHaps[1][i]:
#                 seg = 9
#             if hap[i] == parHaps[0][i] and hap[i] != parHaps[1][i]: 
#                 seg = 0
#             if hap[i] == parHaps[1][i] and hap[i] != parHaps[0][i]:
#                 seg = 1
#         #Compare seg to current hap
#         if seg != 9:
#             if seg == -1:
#                 #This is a weird case -- we basically are saying we don't know what the seg is here because it's not what we expect, but we could use process of elimination to determine it has to be the other one.
#                 if currentSeg != -1 and numMarkers >= reqNumMarkers :
#                     #Do the assignment, mindful of missingness

#                     oldEnd, newStart = getBounds(end, i, fill)
#                     peelDown_assignHaplotype(hap, parHaps, segregation, start, oldEnd, currentSeg) #maybe don't need any args.
#                 currentSeg = -1
#                 start = newStart
#                 end = i
#                 numMarkers = 0
#             else: #Seg not -1 or 9
#                 if currentSeg == -1:
#                     start = i
#                     end = i 
#                     currentSeg = seg
#                     numMarkers = 0
#                     if firstStretch and start < maxGap:
#                         start = 0
#                         firstStretch = False
                
#                 elif seg == currentSeg:
#                     end = i
#                     numMarkers += 1
                
#                 elif seg != currentSeg:
#                     if numMarkers >= reqNumMarkers :
#                         oldEnd, newStart = getBounds(end, i, fill)
#                         peelDown_assignHaplotype(hap, parHaps, segregation, start, oldEnd, currentSeg)
#                     currentSeg = seg
#                     start = newStart
#                     end = i
#                     numMarkers = 0
#     # if currentSeg != -1 and len(hap) - end < maxGap and numMarkers >= reqNumMarkers :
#     if currentSeg != -1 :
#             peelDown_assignHaplotype(hap, parHaps, segregation, start, len(hap), currentSeg)


# @jit(nopython=True)
# def peelDown_correct(hap, parHaps, segregation, fill = 0):
#     #Paramaters
#     reqNumMarkers = 1
#     maxGap = 2*len(hap)
#     #Work out overhang later

#     numMarkers = 0
#     start = 0
#     end = 0
#     currentSeg = -1
    

#     firstStretch = True
#     for i in range(len(hap)):
#         seg = 9
#         #generate seg
#         if i - end > maxGap:
#             seg = -1
#         if hap[i] != 9 and (parHaps[0][i] == 9 or parHaps[1][i] == 9):
#             if currentSeg == 0 and parHaps[0][i] != 9 and hap[i] != parHaps[0][i]:
#                 seg = -1
#             if currentSeg == 1 and parHaps[1][i] != 9 and hap[i] != parHaps[1][i]:
#                 seg = -1
#         elif hap[i] != 9:
#             if hap[i] != parHaps[0][i] and hap[i] != parHaps[1][i]:
#                 seg = -1
#             if hap[i] == parHaps[0][i] and hap[i] != parHaps[1][i]: 
#                 seg = 0
#             if hap[i] == parHaps[1][i] and hap[i] != parHaps[0][i]:
#                 seg = 1
#         #Compare seg to current hap
#         if seg != 9:
#             if seg == -1:
#                 #This is a weird case -- we basically are saying we don't know what the seg is here because it's not what we expect, but we could use process of elimination to determine it has to be the other one.
#                 if currentSeg != -1 and numMarkers >= reqNumMarkers :
#                     #Do the assignment, mindful of missingness

#                     oldEnd, newStart = getBounds(end, i, fill)
#                     peelDown_assignHaplotype(hap, parHaps, segregation, start, oldEnd, currentSeg) #maybe don't need any args.
#                 currentSeg = -1
#                 start = newStart
#                 end = i
#                 numMarkers = 0
#             else: #Seg not -1 or 9
#                 if currentSeg == -1:
#                     start = i
#                     end = i 
#                     currentSeg = seg
#                     numMarkers = 0
#                     if firstStretch and start < maxGap:
#                         start = 0
#                         firstStretch = False
                
#                 elif seg == currentSeg:
#                     end = i
#                     numMarkers += 1
                
#                 elif seg != currentSeg:
#                     if numMarkers >= reqNumMarkers :
#                         oldEnd, newStart = getBounds(end, i, fill)
#                         peelDown_assignHaplotype(hap, parHaps, segregation, start, oldEnd, currentSeg)
#                     currentSeg = seg
#                     start = newStart
#                     end = i
#                     numMarkers = 0
#     if currentSeg != -1 and len(hap) - end < maxGap and numMarkers >= reqNumMarkers :
#             peelDown_assignHaplotype(hap, parHaps, segregation, start, len(hap), currentSeg)

# @jit(nopython=True)
# def peelDown_assignHaplotype(hap, parHaps, segregation, start, end, currentSeg):
#     for j in range(start, end+1):
#         if hap[j] == 9 and parHaps[currentSeg][j] != 9:
#             hap[j] = parHaps[currentSeg][j]
#     segregation.append((currentSeg, start, end))


# @jit(nopython=True, locals = {'end':numba.int64, 'start':numba.int64})
# def getBounds(end, start, fill):
#     if fill == 0: return (end, start)

#     windowSize = (start-end)*fill/2
#     end = end + windowSize
#     start = start - windowSize
#     if end == start: start = start + 1

#     return (end, start)


# def ind_peelUp(ind) :
#     nLoci = len(ind.haplotypes[0])
#     hap0 = Util.jit_haplotypeChunk(nLoci)
#     hap1 = Util.jit_haplotypeChunk(nLoci)

#     for child in ind.offspring:
#         fillInParentFromOffspring(hap0, hap1, child.haplotypes[ind.gender], child.segregation[ind.gender])
#     filInIfMissing(ind.haplotypes[0], hap0.callHaplotype())
#     filInIfMissing(ind.haplotypes[1], hap1.callHaplotype())


#     ind_fillInGenotypesFromPhase(ind)

# @jit(nopython=True)
# def fillInParentFromOffspring(hap0, hap1, refHap, segregation) :
#     for i in range(hap0.nLoci) :
#         seg = segregation.getSeg(i)
#         if seg == 0:
#             hap0.counts[i] += 1
#             hap0.ones[i] += refHap[i]
#         if seg == 1:
#             hap1.counts[i] += 1
#             hap1.ones[i] += refHap[i]



#     #####################################
#     #####################################
#     ####                            #####
#     ####  ProbPedigree Imputation   #####
#     ####                            #####
#     #####################################
#     #####################################


# def ind_peelDown_prob(ind, threshold = .99, peelUp = False):

#     #If they have a sire, they have to have a dam.
#     if ind.sire is not None and ind.dam is not None:
#         # ind.segregation[0] = jit_segregation()
#         # peelDown_fimpute(ind.haplotypes[0], ind.sire.haplotypes, ind.segregation[0])
#         peelDown_prob(ind.haplotypes, ind.sire.haplotypes, ind.dam.haplotypes, threshold = threshold, peelUp = peelUp)
#         ind_align(ind)
# @jit(nopython=True)
# def peelDown_prob(haps, sireHaps, damHaps, threshold, peelUp = False):
#     # 1) create pointSeg 
#     # 2) do a forward-backward on the pentrance
#     # 3) find loci that have a clear segregation, and call them.
#     pointSeg = getPeelDownPointSeg(haps, sireHaps, damHaps)
#     nLoci = len(haps[0])

#     rate = 1.0/nLoci
#     values = loopyPeelingForwardBack(pointSeg, rate)
#     # print(values)
#     #Now set values.
#     for parIndex in range(2):
#         if parIndex == 0:
#             parHaps = sireHaps
#         else:
#             parHaps = damHaps
#         callChildHaps(haps[parIndex], parIndex, values, parHaps, threshold)
        
#     #Currently not using this. Need a good argument for a use case I think.
#     if peelUp:
#         for parIndex in range(2):
#             if parIndex == 0:
#                 parHaps = sireHaps
#             else:
#                 parHaps = damHaps
#             calledHaps = callParentHaps(haps[parIndex], parIndex, values, parHaps, threshold)

#             filInIfMissing(parHaps[0], calledHaps[0])
#             filInIfMissing(parHaps[1], calledHaps[1])


# @jit(nopython=True)
# def callChildHaps(hap, parIndex, values, parHaps, threshold):
#     nLoci = len(hap)
#     thresh0 = 1-threshold
#     thresh1 = threshold
#     for i in range(nLoci):
#         if hap[i] == 9:
#             if parIndex == 0:
#                 #Val is probablity of hap 1
#                 val = values[i,2] + values[i,3]
#             else:
#                 val = values[i,1] + values[i,3]
#             # print(thresh0, val)
#             if val < thresh0:
#                 hap[i] = parHaps[0][i]
#             elif val > thresh1:
#                 hap[i] = parHaps[1][i]

# @jit(nopython=True)
# def callParentHaps(hap, parIndex, values, parHaps, threshold):
#     nLoci = len(hap)
#     thresh0 = 1-threshold
#     thresh1 = threshold
#     calledHaps = np.full((nLoci, 2), 9, dtype = np.int8)

#     for i in range(nLoci):
#         if hap[i] != 9:
#             if parIndex == 0:
#                 #Val is probablity of hap 1
#                 val = values[i,2] + values[i,3]
#             else:
#                 val = values[i,1] + values[i,3]
#             if val < thresh0:
#                 if parHaps[0][i] == 9:
#                     calledHaps[0][i] = hap[i]

#             elif val > thresh1:
#                 if parHaps[1][i] == 9:
#                     calledHaps[1][i] = hap[i]                
#     return calledHaps

# @jit(nopython=True)
# def getPeelDownPointSeg(haps, sireHaps, damHaps):

#     e = 0.01
#     ei = 1-e
#     #Same as peeling (I think).
#     # pp, pm, mp, mm
#     nLoci = len(haps[0])
#     pointSeg = np.full((nLoci, 4), 1, dtype = np.float32)

#     for parIndex in range(2):
#         if parIndex == 0:
#             parHaps = sireHaps
#         else:
#             parHaps = damHaps
#         for i in range(nLoci):
#             #Not hugely sure what to do if one is missing and the other is not.
#             #Will see if this is a problem later.
#             if haps[parIndex][i] != 9 and parHaps[0][i] != 9 and parHaps[1][i] !=9:
#                 if parHaps[0][i] == parHaps[1][i]:
#                     #Nothing to do if the parent is homozygous.
#                     pass                    
#                 else:
#                     #Hap 0 is the right fit.
#                     if parHaps[0][i] == haps[parIndex][i] and parHaps[1][i] != haps[parIndex][i]:
#                         hap0Val = ei
#                         hap1Val = e
#                     #hap 1 is the right fit.
#                     if parHaps[1][i] == haps[parIndex][i] and parHaps[0][i] != haps[parIndex][i]:
#                         hap0Val = e
#                         hap1Val = ei
#                     #PaternalHaplotype
#                     if parIndex == 0:
#                         pointSeg[i,0] *= hap0Val # 0,0
#                         pointSeg[i,1] *= hap0Val # 0,1
#                         pointSeg[i,2] *= hap1Val # 1,0
#                         pointSeg[i,3] *= hap1Val # 1,1
#                     #MaternalHaplotype
#                     if parIndex == 1:
#                         pointSeg[i,0] *= hap0Val # 0,0
#                         pointSeg[i,1] *= hap1Val # 0,1
#                         pointSeg[i,2] *= hap0Val # 1,0
#                         pointSeg[i,3] *= hap1Val # 1,1
#     return pointSeg
# @jit(nopython=True, locals={'e': float32, 'e2':float32, 'e1e':float32, 'e2i':float32})
# def loopyPeelingForwardBack(pointSeg, rate):
#     #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
#     e = rate
#     e2 = e**2
#     e1e = e*(1-e)
#     e2i = 1.0 - e2

#     nLoci = pointSeg.shape[0] 

#     seg = np.full(pointSeg.shape, .25, dtype = np.float32)
#     for i in range(nLoci):
#         for j in range(4):
#             seg[i,j] = pointSeg[i,j]

#     tmp = np.full((4), 0, dtype = np.float32)
#     new = np.full((4), 0, dtype = np.float32)

#     prev = np.full((4), .25, dtype = np.float32)
#     for i in range(1, nLoci):
#         for j in range(4):
#             tmp[j] = prev[j]*pointSeg[i-1,j]
        
#         sum_j = 0
#         for j in range(4):
#             sum_j += tmp[j]
#         for j in range(4):
#             tmp[j] = tmp[j]/sum_j

#         # !                  fm  fm  fm  fm 
#         # !segregationOrder: pp, pm, mp, mm

#         new[0] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
#         new[1] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
#         new[2] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
#         new[3] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

#         # tmp = tmp/np.sum(tmp)
#         # new = e2i*tmp + e2 + e1e*(tmp[0] + tmp[3])*same + e1e*(tmp[1] + tmp[2])*diff       

#         for j in range(4):
#             seg[i,j] *= new[j]
#         # seg[:,i] *= new
#         prev = new

#     prev = np.full((4), .25, dtype = np.float32)
#     for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
#         for j in range(4):
#             tmp[j] = prev[j]*pointSeg[i+1,j]
        
#         sum_j = 0
#         for j in range(4):
#             sum_j += tmp[j]
#         for j in range(4):
#             tmp[j] = tmp[j]/sum_j

#         new[0] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
#         new[1] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
#         new[2] = e2 + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
#         new[3] = e2 + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

#         for j in range(4):
#             seg[i,j] *= new[j]
#         prev = new

#     for i in range(nLoci):
#         sum_j = 0
#         for j in range(4):
#             sum_j += seg[i,j]
#         for j in range(4):
#             seg[i,j] = seg[i,j]/sum_j

#     return(seg)





#     #####################################
#     #####################################
#     ####                            #####
#     ####     PeelDown Imputation    #####
#     ####                            #####
#     #####################################
#     #####################################



# def imputeFromChildren(ind, cutoff = 0.9999) :

#     nLoci = len(ind.haplotypes[0])
#     ones = np.full((nLoci, 2), 0, np.int64)
#     counts = np.full((nLoci, 2), 0, np.int64)

#     matLib = np.transpose(np.array(ind.haplotypes))

#     if len(ind.offspring) > 0:
#         for child in ind.offspring:
#             childHap = None
#             if child.sire is ind:
#                 childHap = child.haplotypes[0]
#             if child.dam is ind:
#                 childHap = child.haplotypes[1]
#             if childHap is None:
#                 print("Something bad happened. Child of parent does not have parent as a parent")

#             pointEstimates = getPointSegs(childHap, matLib, error = 0.01)

#             rate = 1.0/nLoci

#             probs = haploidForwardBack(pointEstimates, rate)
#             calledHap = callReverseProbs(probs, childHap, cutoff = cutoff)

#             addToCountsIfNotMissing_2D(calledHap, ones, counts)
        
#         for e in range(2):
#             calledHap = callCounts(ones[:,e], counts[:,e], threshold = .95)
#             filInIfMissing(orig = ind.haplotypes[e], new = calledHap)

# @jit(nopython=True)
# def callCounts(ones, counts, threshold):
#     nLoci = len(ones)
#     calledHap = np.full(nLoci, 9, np.int8)
#     for i in range(nLoci):
#         if ones[i] > counts[i]*threshold:
#             calledHap[i] = 1
#         if ones[i] < counts[i]*(1-threshold):
#             calledHap[i] = 0
#     return calledHap

# @jit(nopython=True)
# def callReverseProbs(probs, refHap, cutoff):
#     nLoci, nHaps = probs.shape
#     hap = np.full(probs.shape, 9, dtype = np.int8)
#     #Work on calling first, and then use that. Don't use dosages.
#     for i in range(nLoci):
#         for j in range(nHaps):
#             score = probs[i, j]
#             if score > cutoff:
#                 hap[i, j] = refHap[i]
#     return hap

# @jit(nopython=True)
# def addToCountsIfNotMissing_2D(haplotypes, ones, counts):
#     nLoci, nHaps = haplotypes.shape
#     for i in range(nLoci):
#         for j in range(nHaps):
#             if haplotypes[i, j] != 9:
#                 counts[i,j] += 1
#                 ones[i,j] += haplotypes[i,j]


#     #####################################
#     #####################################
#     ####                            #####
#     ####  HaploidFamily Imputation  #####
#     ####                            #####
#     #####################################
#     #####################################

# def getAncestors(ind, depth):
#     if ind is None:
#         return []
#     if depth == 0:
#         return [ind]
#     else:
#         ancestors = []
#         if ind.sire is not None:
#             ancestors += getAncestors(ind.sire, depth-1)
#         if ind.dam is not None:
#             ancestors += getAncestors(ind.dam, depth-1)
#         return ancestors

# def imputeFromAncestors(ind, depth = 1, cutoff = 0.9999):

#     #sire
#     if ind.sire is not None:
#         haploidFamilyHMM(ind.haplotypes[0], getAncestors(ind.sire, depth=depth), cutoff = cutoff)
#         ind_align(ind)
#     if ind.dam is not None:
#         haploidFamilyHMM(ind.haplotypes[1], getAncestors(ind.dam, depth=depth), cutoff = cutoff)
#         ind_align(ind)


# def haploidFamilyHMM(hap, individuals, cutoff = 0.9):

#     hapLib = []
#     for ind in individuals:
#         for e in range(2):
#             # if np.mean(ind.haplotypes[e] == 9) <.1:
#             hapLib.append(ind.haplotypes[e])    

#     if len(hapLib) >0 :
#         matLib = np.transpose(np.array(hapLib))
        

#         #Maybe I'm being silly here...
#         nLoci, nHaps = matLib.shape

#         pointEstimates = getPointSegs(hap, matLib, error = 0.01)

#         rate = 1.0/nLoci

#         probs = haploidForwardBack(pointEstimates, rate)
#         # print("collapsed")
#         # print(probs[0:30, :])

#         calledHap = callProbs(probs, matLib, cutoff = cutoff)

#         filInIfMissing(hap, calledHap)

# # @jit(nopython=True)
# # def callProbs_dosages(probs, matLib, cutoff):
# #     nLoci, nHaps = matLib.shape
# #     hap = np.full(nLoci, 9, dtype = np.int8)
# #     #Work on calling first, and then use that. Don't use dosages.
# #     for i in range(nLoci):
# #         score = 0
# #         weight = 0
# #         for j in range(nHaps):
# #             val = matLib[i, j]
# #             if val != 9:
# #                 score += matLib[i, j]*probs[i,j]
# #                 weight += probs[i,j]
# #         if weight > 0:
# #             score = score/weight
# #             if score > cutoff:
# #                 hap[i] = 1
# #             if score <= 1-cutoff:
# #                 hap[i] = 0
# #     return hap
# @jit(nopython=True)
# def callProbs(probs, matLib, cutoff):
#     nLoci, nHaps = matLib.shape
#     hap = np.full(nLoci, 9, dtype = np.int8)
#     #Work on calling first, and then use that. Don't use dosages.
#     for i in range(nLoci):
#         for j in range(nHaps):
#             score = probs[i, j]
#             if score > cutoff:
#                 hap[i] = matLib[i, j]
#     return hap
# @jit(nopython=True)
# def getPointSegs(hap, haps, error):
#     nLoci, nHaps = haps.shape
#     pointEstimates = np.full(haps.shape, .5, dtype = np.float32)
#     for i in range(nLoci):
#         if hap[i] != 9:
#             include = True
#             for j in range(nHaps):
#                 if haps[i, j] == 9:
#                     include = False
#                     break
#             if include:
#                 for j in range(nHaps):
#                     if haps[i, j] == 9:
#                         pointEstimates[i, j] = .5
#                     if hap[i] == haps[i, j]:
#                         pointEstimates[i, j] = 1 - error
#                     else:
#                         pointEstimates[i, j] = error
#     return pointEstimates

# @jit(nopython=True, locals={'e': float32, 'e1':float32})
# def haploidForwardBack(pointSeg, rate):
#     #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
#     e = rate
#     e1 = 1-e
#     nLoci = pointSeg.shape[0] 
#     nHaps = pointSeg.shape[1]

#     seg = np.full(pointSeg.shape, .25, dtype = np.float32)
#     for i in range(nLoci):
#         for j in range(nHaps):
#             seg[i,j] = pointSeg[i,j]

#     tmp = np.full(nHaps, 0, dtype = np.float32)
#     new = np.full(nHaps, 0, dtype = np.float32)

#     prev = np.full(nHaps, .25, dtype = np.float32)
    
#     for i in range(1, nLoci):
#         for j in range(nHaps):
#             tmp[j] = prev[j]*pointSeg[i-1,j]
        
#         sum_j = 0
#         for j in range(nHaps):
#             sum_j += tmp[j]
#         for j in range(nHaps):
#             tmp[j] = tmp[j]/sum_j

#         # !                  fm  fm  fm  fm 
#         # !segregationOrder: pp, pm, mp, mm

#         for j in range(nHaps):
#             new[j] = e + e1*tmp[j]

#         # tmp = tmp/np.sum(tmp)
#         # new = e2i*tmp + e2 + e1e*(tmp[0] + tmp[3])*same + e1e*(tmp[1] + tmp[2])*diff       

#         for j in range(nHaps):
#             seg[i,j] *= new[j]
#         # seg[:,i] *= new
#         prev = new

#     prev = np.full((nHaps), .25, dtype = np.float32)
#     for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
#         for j in range(nHaps):
#             tmp[j] = prev[j]*pointSeg[i+1,j]
        
#         sum_j = 0
#         for j in range(nHaps):
#             sum_j += tmp[j]
#         for j in range(nHaps):
#             tmp[j] = tmp[j]/sum_j
        
#         for j in range(nHaps):
#             new[j] = e + e1*tmp[j]

#         for j in range(nHaps):
#             seg[i,j] *= new[j]
#         prev = new

#     for i in range(nLoci):
#         sum_j = 0
#         for j in range(nHaps):
#             sum_j += seg[i,j]
#         for j in range(nHaps):
#             seg[i,j] = seg[i,j]/sum_j

#     return(seg)

