from .tinyhouse import Pedigree
from .tinyhouse import InputOutput
from .tinyhouse import ProbMath

from .Imputation import ProbPhasing
from .Imputation import ParticlePhasing
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



    #####################################
    #####################################
    ####                            #####
    ####    Imputation Programs     #####
    ####                            #####
    #####################################
    #####################################


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
    
    core_impute_parser.add_argument('-phase', action='store_true', required=False, help='Flag to run the phasing algorithm.')
   


    core_impute_parser.add_argument('-cutoff',default=.95, required=False, type=float, help='Genotype calling threshold.')
    core_impute_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles.')
    core_impute_parser.add_argument('-length', default=1.0, required=False, type=float, help='Estimated length of the chromosome in Morgansa. [Default 1.00]')

    return InputOutput.parseArgs("AlphaImpute", parser)

def writeOutResults(pedigree, args):
    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        # for ind in pedigree:
        #     ind.peeling_view.setGenotypesAll(.7)

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


@profile
def main():
    
    # Set up arguments and pedigree
    args = getArgs()
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)

    # Fill in haplotypes and phase
    setupImputation(pedigree)

    if args.phase:
        hd_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9) > .9]
        print(len(hd_individuals), "Sent to phasing")
        ParticlePhasing.phase_individuals(hd_individuals, pedigree)
        # ProbPhasing.run_phaseHD(pedigree)

    # Run family based phasing.
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = .8)

    # Write out results
    startTime = datetime.datetime.now()

    writeOutResults(pedigree, args)
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()



if __name__ == "__main__":
    main()