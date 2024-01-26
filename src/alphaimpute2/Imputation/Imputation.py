from numba import jit


import numpy as np


def ind_fillInGenotypesFromPhase(ind):
    fillInGenotypesFromPhase(ind.genotypes, ind.haplotypes[0], ind.haplotypes[1])


def ind_align(ind):
    fillInPhaseFromGenotypes(ind.haplotypes[0], ind.genotypes)
    fillInPhaseFromGenotypes(ind.haplotypes[1], ind.genotypes)

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
        if phase[i] == 9:
            if geno[i] == 0:
                phase[i] = 0
            if geno[i] == 2:
                phase[i] = 1


def ind_resetPhasing(ind):
    ind.haplotypes[0][:] = 9
    ind.haplotypes[1][:] = 9
    fillInPhaseFromGenotypes(ind.haplotypes[0], ind.genotypes)
    fillInPhaseFromGenotypes(ind.haplotypes[1], ind.genotypes)
    ind_randomlyPhaseRandomPoint(ind)


def ind_randomlyPhaseRandomPoint(ind):
    maxLength = len(ind.genotypes)
    midpoint = np.random.normal(maxLength / 2, maxLength / 10)
    while midpoint < 0 or midpoint > maxLength:
        midpoint = np.random.normal(maxLength / 2, maxLength / 10)
    midpoint = int(midpoint)
    randomlyPhaseMidpoint(ind.genotypes, ind.haplotypes, midpoint)


def ind_randomlyPhaseMidpoint(ind, midpoint=None):
    if midpoint is None:
        midpoint = int(len(ind.genotypes) / 2)
    randomlyPhaseMidpoint(ind.genotypes, ind.haplotypes, midpoint)


@jit(nopython=True)
def randomlyPhaseMidpoint(geno, phase, midpoint):
    index = 0
    e = 1
    changed = False
    while not changed:
        if geno[midpoint + index * e] == 1:
            phase[0][midpoint + index * e] = 0
            phase[1][midpoint + index * e] = 1
            changed = True
        e = -e
        if e == -1:
            index += 1
        if index >= midpoint:
            changed = True

    #   Random Haplotype math?   #


def ind_getDosages(ind, maf=None):
    nLoci = len(ind.genotypes)

    if maf is None:
        maf = np.full(nLoci, 0.5)
    ind.genotypeDosages = np.zeros(nLoci)
    ind.haplotypeDosages = (np.zeros(nLoci), np.zeros(nLoci))

    if ind.sire is not None:
        if ind.sire.haplotypeDosages is None:
            ind_getDosages(ind.sire)
        sireHaps = ind.sire.haplotypeDosages
    else:
        sireHaps = (maf, maf)
    getHaplotypeDosage(
        ind.haplotypes[0], ind.haplotypeDosages[0], sireHaps, ind.segregation[0]
    )

    if ind.dam is not None:
        if ind.dam.haplotypeDosages is None:
            ind_getDosages(ind.dam)
        damHaps = ind.dam.haplotypeDosages
    else:
        damHaps = (maf, maf)
    getHaplotypeDosage(
        ind.haplotypes[1], ind.haplotypeDosages[0], damHaps, ind.segregation[1]
    )

    getGenotypeDosageFromHaplotype(
        ind.genotypes, ind.genotypeDosages, ind.haplotypeDosages
    )


def ind_assignOrigin(ind):
    nLoci = len(ind.genotypes)
    ind.hapOfOrigin = (np.zeros(nLoci, dtype=np.int32), np.zeros(nLoci, dtype=np.int32))
    ind.hapOfOrigin[0][:] = int(ind.idx) * 2 + 0
    ind.hapOfOrigin[1][:] = int(ind.idx) * 2 + 1

    if ind.sire is not None:
        if ind.sire.hapOfOrigin is None:
            ind_assignOrigin(ind.sire)
        getHaplotypeOfOrigin(
            ind.hapOfOrigin[0], ind.sire.hapOfOrigin, ind.segregation[0]
        )

    if ind.dam is not None:
        if ind.dam.hapOfOrigin is None:
            ind_assignOrigin(ind.dam)
        getHaplotypeOfOrigin(
            ind.hapOfOrigin[1], ind.dam.hapOfOrigin, ind.segregation[1]
        )


@jit(nopython=True)
def getHaplotypeOfOrigin(hapOrigin, parOrigins, segregation):
    for i in range(len(hapOrigin)):
        seg = segregation.getSeg(i)
        if seg == -1:
            pass
            # hapOrigin[i] = -1
        else:
            hapOrigin[i] = parOrigins[seg][i]


@jit(nopython=True)
def getSegregation(nLoci, segregation):
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
            dosages[i] = parDosages[0][i] * (1 - seg) + parDosages[1][i] * seg


@jit(nopython=True)
def getGenotypeDosageFromHaplotype(geno, dosages, hapDosages):
    for i in range(len(geno)):
        if geno[i] != 9:
            dosages[i] = geno[i]
        else:
            dosages[i] = hapDosages[0][i] + hapDosages[1][i]


def pedCompareGenotypes(refPed, truePed):
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
                errors += np.sum(
                    np.logical_and(
                        refInd.genotypes != 9, refInd.genotypes != trueInd.genotypes
                    )
                )
    return (y / nLoci, 1 - errors / y)


def pedComparePhase(refPed, truePed):
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

                error1 = np.sum(
                    np.logical_and(
                        refInd.haplotypes[0] != 9,
                        refInd.haplotypes[0] != trueInd.haplotypes[0],
                    )
                )
                error1 += np.sum(
                    np.logical_and(
                        refInd.haplotypes[1] != 9,
                        refInd.haplotypes[1] != trueInd.haplotypes[1],
                    )
                )

                error2 = np.sum(
                    np.logical_and(
                        refInd.haplotypes[0] != 9,
                        refInd.haplotypes[0] != trueInd.haplotypes[1],
                    )
                )
                error2 += np.sum(
                    np.logical_and(
                        refInd.haplotypes[1] != 9,
                        refInd.haplotypes[1] != trueInd.haplotypes[0],
                    )
                )

                errors += min(error1, error2)
    return (y / nLoci, 1 - errors / y)


def pedCompareDosages(refPed, truePed):
    n = 0
    cor = 0
    for idx in refPed.individuals:
        if idx in truePed.individuals:
            refInd = refPed.individuals[idx]
            trueInd = truePed.individuals[idx]
            ind_getDosages(refInd)
            cor += np.corrcoef(refInd.genotypeDosages, trueInd.genotypes)
            n += 1
    return cor / n


# Exclude is trying to prevent gratuitous backtracking.
def getRelatives(focal, maxDist, excludeInds, direction="both"):
    val = set()

    print(focal)
    print(excludeInds)
    if direction in ("both", "up") and focal.sire is not None:
        if focal.sire not in excludeInds:
            val.add(focal.sire)
    if direction in ("both", "up") and focal.dam is not None:
        if focal.dam not in excludeInds:
            val.add(focal.dam)

    if direction in ("both", "down"):
        for child in focal.offspring:
            if child not in excludeInds:
                val.add(child)

    newExclude = excludeInds.union(val)
    if maxDist > 0:
        for ind in val:
            val = val.union(getRelatives(ind, maxDist - 1, newExclude))
    return val
