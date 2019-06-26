from .tinyhouse import Pedigree
from .tinyhouse import InputOutput
from .tinyhouse import ProbMath

from .Imputation import ProbPhasing
from .Imputation import HeuristicBWImpute
from .Imputation import Heuristic_Peeling
from .Imputation import ImputationIndividual
from .Imputation import Imputation

from .Imputation import FamilyParentPhasing


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
        Imputation.ind_align(ind)

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
    core_impute_parser.add_argument('-maxthreads',default=1, required=False, type=int, help='Number of threads to use. Default: 1.')
    core_impute_parser.add_argument('-binaryoutput', action='store_true', required=False, help='Flag to write out the genotypes as a binary plink output.')
   


    core_impute_parser.add_argument('-cutoff',default=.95, required=False, type=float, help='Genotype calling threshold.')
    core_impute_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles.')

    return InputOutput.parseArgs("AlphaImpute", parser)



@profile
def main():
    
    ### Setup
    startTime = datetime.datetime.now()

    args = getArgs()
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)

    # Fill in haplotypes from genotypes. Fill in genotypes from phase.
    setupImputation(pedigree)

    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = .1)


    # FamilyParentPhasing.phaseFounders(pedigree)

    # if False:
    #     # Perform initial imputation + phasing before sending it to the phasing program to get phased.
    #     # The imputeBeforePhasing just runs a peel-down, with ancestors included.        
    #     print("Performing initial pedigree imputation")
    #     imputeBeforePhasing(pedigree)
    #     print("Initial pedigree imputation finished.", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()
        
    #     # Phasing
    #     print("HD Phasing started.")
    #     if not args.no_phase:
    #         phaseHD(pedigree)
    #     print("HD Phasing", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    #     # Imputation.
    #     for genNum, gen in enumerate(pedigree.generations):
    #         # This runs a mix of pedigree based and population based imputation algorithms to impute each generation based on phased haplotypes.
    #         print("Generation:",  genNum)
    #         imputeGeneration(gen, pedigree, args, genNum)
    #         print("Generation ", genNum, datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    #Write out.
    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        for ind in pedigree:
            ind.peeling_view.setGenotypesAll(.1)

        pedigree.writeGenotypes(args.out + ".genotypes")
        pedigree.writePhase(args.out + ".phase")

        # for ind in pedigree:
        #     ind.peeling_view.setGenotypesPenetrance(.1)
        # pedigree.writeGenotypes(args.out + ".penetrance")

        # for ind in pedigree:
        #     ind.peeling_view.setGenotypesFromPeelingData(True, False, False, .1)

        # pedigree.writeGenotypes(args.out + ".anterior")

        # pedigree.writeSegregation(args.out + ".seg.0", 0)
        # pedigree.writeSegregation(args.out + ".seg.1", 1)
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()



if __name__ == "__main__":
    main()