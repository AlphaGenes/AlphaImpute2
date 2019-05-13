from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import Imputation
from .Imputation import PedigreeImputation
from .Imputation import ProbPhasing
from .Imputation import HeuristicBWImpute
from .Imputation import Heuristic_Peel_Up
from .Imputation import Heuristic_Peeling
from .Imputation import Heuristic_Peeling_Careful
from .Imputation import ImputationIndividual

import datetime
import argparse
import numpy as np

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
        ind.setupIndividual()
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[0], ind.genotypes)
        Imputation.fillInPhaseFromGenotypes(ind.haplotypes[1], ind.genotypes)
        Imputation.ind_fillInGenotypesFromPhase(ind)
        # if ind.isFounder() :
        #     Imputation.ind_randomlyPhaseMidpoint(ind)

def imputeBeforePhasing(pedigree):
    # Idea here: Use pedigree information to pre-phase individuals before phasing them.
    # We should probably use both parents + ancestors to do the imputation, and consider peeling up as well as peeling down.
    # Right now, we just perform peeling.

    # PedigreeImputation.performPeeling(pedigree, fill = .99, ancestors = False)

    for ind in pedigree:
        Heuristic_Peeling.setSegregation(ind)
        Heuristic_Peeling.heuristicPeelDown(ind)



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


    #####################################
    #####################################
    ####                            #####
    ####    Imputation              #####
    ####                            #####
    #####################################
    #####################################



def getArgs() :
    parser = argparse.ArgumentParser(description='')
    core_parser = parser.add_argument_group("Core arguments")
    core_parser.add_argument('-out', required=True, type=str, help='The output file prefix.')
    InputOutput.addInputFileParser(parser)
    
    core_impute_parser = parser.add_argument_group("Impute options")
    core_impute_parser.add_argument('-no_impute', action='store_true', required=False, help='Flag to read in the files but not perform imputation.')
    core_impute_parser.add_argument('-no_phase', action='store_true', required=False, help='Flag to not do HD phasing initially.')
    core_impute_parser.add_argument('-maxthreads',default=1, required=False, type=int, help='Number of threads to use. Default: 1.')
    core_impute_parser.add_argument('-binaryoutput', action='store_true', required=False, help='Flag to write out the genotypes as a binary plink output.')

    core_impute_parser.add_argument('-peelup', action='store_true', required=False, help='Flag to peel up.')
    core_impute_parser.add_argument('-peeldown', action='store_true', required=False, help='Flag to peel down.')
    core_impute_parser.add_argument('-peeldown_multi', action='store_true', required=False, help='Flag to peel down with the fancier peeling.')

    return InputOutput.parseArgs("AlphaFamImpute", parser)



@profile
def main():
    
    ### Setup
    startTime = datetime.datetime.now()

    args = getArgs()
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)
    # Fill in haplotypes from genotypes. Fill in genotypes from phase.
    setupImputation(pedigree)


    for ind in pedigree:
        ind.toJit().setValueFromGenotypes(ind.penetrance)
    cutoffs = [.99, .99, .98, .97, .95]
    for cycle in range(len(cutoffs)):

        for ind in pedigree:
            Heuristic_Peeling_Careful.setSegregation(ind, cutoff = cutoffs[cycle])
            Heuristic_Peeling_Careful.heuristicPeelDown(ind)

        for ind in reversed(pedigree):
            # Heuristic_Peel_Up.singleLocusPeelUp(ind)
            Heuristic_Peeling_Careful.setSegregation(ind, cutoff = cutoffs[cycle])
            Heuristic_Peeling_Careful.HeuristicPeelUp(ind)

        print("Performing initial pedigree imputation")


        for ind in pedigree:
            ind.setGenotypesAll()

        pedigree.writeGenotypes(args.out + ".genotypes." + str(cycle))

    # for ind in pedigree:
    #     ind.setGenotypesPenetrance()
    # print("Read in and initial setup", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    # if not args.no_impute: 
    if False:
        # Perform initial imputation + phasing before sending it to the phasing program to get phased.
        # The imputeBeforePhasing just runs a peel-down, with ancestors included.        
        print("Performing initial pedigree imputation")
        imputeBeforePhasing(pedigree)
        print("Initial pedigree imputation finished.", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()
        
        # Phasing
        print("HD Phasing started.")
        if not args.no_phase:
            phaseHD(pedigree)
        print("HD Phasing", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

        # Imputation.
        for genNum, gen in enumerate(pedigree.generations):
            # This runs a mix of pedigree based and population based imputation algorithms to impute each generation based on phased haplotypes.
            print("Generation:",  genNum)
            imputeGeneration(gen, pedigree, args, genNum)
            print("Generation ", genNum, datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    #Write out.
    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        pedigree.writeGenotypes(args.out + ".genotypes")
        # pedigree.writeGenotypes_prefil(args.out + ".genotypes")
        pedigree.writePhase(args.out + ".phase")
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()



if __name__ == "__main__":
    main()