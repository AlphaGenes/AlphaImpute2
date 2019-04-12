import numpy as np
from math import floor
import random
from . import Imputation
from . import PedigreeImputation
from . import ProbPhasing
from . import HeuristicBWImpute


try:
    profile
except:
    def profile(x): 
        return x



    #####################################
    #####################################
    ####                            #####
    #### Core tinyImpute Imputation #####
    ####                            #####
    #####################################
    #####################################





def setupImputation(pedigree):
    for ind in pedigree :
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[0], ind.genotypes)
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[1], ind.genotypes)
        Imputation.ind_fillInGenotypesFromPhase(ind)
        if ind.isFounder() :
            Imputation.ind_randomlyPhaseMidpoint(ind)

def imputeBeforePhasing(pedigree):
    # Idea here: Use pedigree information to pre-phase individuals before phasing them.
    # We should probably use both parents + ancestors to do the imputation, and consider peeling up as well as peeling down.
    # Right now, we just perform peeling.
    
    PedigreeImputation.performPeeling(pedigree, fill = .99, ancestors = True)

def phaseHD(pedigree):

    phasingInfo = ProbPhasing.PhasingInformation()
    indList = [ind for ind in pedigree if ind.initHD]

    print("Phasing HD individuals: ", len(indList))
    markers = np.array([i for i in range(pedigree.nLoci)], dtype = np.int64)
    phasingInfo.addIndividuals(indList, markers)

    runPhasing(phasingInfo, nHets = [20, 30, 40, 50], finalNHet = 50)

@profile
def imputeGeneration(gen, pedigree, args, genNum):
    # peel down to this generation.
    # *peel up to this generation
    # phase this generation using all individuals [Maybe looking at doing this in a chip-specific way]
    # haploid imputation on this generation to fill in the remaining missingness.
    
    #Peel down to generation

    PedigreeImputation.performPeeling(gen, fill = .99, ancestors = True, peelUp=True)
    PedigreeImputation.performPeeling(gen, fill = .9, ancestors = True)
    PedigreeImputation.performPeeling(gen, fill = .8, ancestors = True)
    PedigreeImputation.performPeeling(gen, fill = .5)
    # if genNum == 0:
    if True:
        runLDPhasing(indList = gen, pedigree = pedigree)

        refIndList = None
        refIndList = [ind for ind in pedigree if ind.initHD]
        if len(refIndList) > 10000:
            #Do a subset.
            refIndList = random.sample(refIndList, 10000)

        HeuristicBWImpute.impute(indList = gen, referencePanel = refIndList)


    #####################################
    #####################################
    ####                            #####
    ####    Imputation Programs     #####
    ####                            #####
    #####################################
    #####################################




# global_PhaseAccuracy = None 
# global_GenoAccuracy = None


# def assessAccuracy(label, pedigree) :
#     if pedigree.truePed is not None:
#         currentPhaseAcc = Imputation.pedCompareGenotypes(pedigree, pedigree.truePed)
#         currentGenoAcc = Imputation.pedComparePhase(pedigree, pedigree.truePed)

#         print(label, currentPhaseAcc)
#         print(label, currentGenoAcc)




@profile
def runPhasing(phasingInfo, nHets, finalNHet) :
    for nHet in nHets:
        ProbPhasing.phaseHD(phasingInfo, nHet, setHap = False)
    ProbPhasing.phaseHD(phasingInfo, finalNHet, setHap = True)
    phasingInfo.alignIndividuals()
    

@profile
def runLDPhasing(pedigree, indList = None):
    useRefPanel = True
    if indList is None: 
        indList = [ind for ind in pedigree]
        useRefPanel = False

    phasingInfo = ProbPhasing.PhasingInformation(useReferenceOnly = useRefPanel)
    # missingness = pedigree.getMissingness()
    # markers = np.array([i for i in range(pedigree.nLoci) if missingness[i] < .05], dtype = np.int64)

    markers = np.array([i for i in range(pedigree.nLoci)], dtype = np.int64)
    phasingInfo.addIndividuals(indList, markers)

    refIndList = None
    if useRefPanel:
        #Let's use initHD as a filter for now. 
        #Will want to think more seriously about this later.

        refIndList = [ind for ind in pedigree if ind.initHD]
        if len(refIndList) > 10000:
            #Do a subset.
            refIndList = random.sample(refIndList, 10000)
        phasingInfo.addReference(refIndList)


    runPhasing(phasingInfo, nHets = [], finalNHet = 20)

