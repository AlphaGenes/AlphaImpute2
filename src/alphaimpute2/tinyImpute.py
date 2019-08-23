from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import ParticlePhasing
from .Imputation import ParticleImputation
from .Imputation import Heuristic_Peeling
from .Imputation import ImputationIndividual
from .Imputation import Imputation


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
    core_impute_parser.add_argument('-popimpute', action='store_true', required=False, help='Flag to run the phasing algorithm.')
    core_impute_parser.add_argument('-pedimpute', action='store_true', required=False, help='Flag to run the phasing algorithm.')
   


    core_impute_parser.add_argument('-cutoff',default=.95, required=False, type=float, help='Genotype calling threshold.')
    core_impute_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles.')


    core_impute_parser.add_argument('-n_phasing_particles',default=80, required=False, type=int, help='Number of phasing particles. Defualt: 40.')
    core_impute_parser.add_argument('-n_phasing_cycles',default=5, required=False, type=int, help='Number of phasing cycles. Default: 4')

    core_impute_parser.add_argument('-n_imputation_particles',default=100, required=False, type=int, help='Number of imputation particles. Defualt: 100.')

    return InputOutput.parseArgs("AlphaImpute", parser)

def writeOutResults(pedigree, args):
    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        pedigree.writeGenotypes(args.out + ".genotypes")
        # pedigree.writePhase(args.out + ".phase")
        
        # writeGenoProbs(pedigree, lambda ind: ind.phasing_view.forward, args.out + ".forward")
        # writeGenoProbs(pedigree, lambda ind: ind.phasing_view.backward, args.out + ".backward")
        # writeGenoProbs(pedigree, lambda ind: ind.phasing_view.penetrance, args.out + ".penetrance")

        # with open(args.out + ".called", 'w+') as f:
        #     for ind in pedigree:
        #         f.write(ind.idx + ' ' + ' '.join(map(str, ind.phasing_view.called_genotypes)) + '\n')


def writeGenoProbs(pedigree, genoProbFunc, outputFile):
    with open(outputFile, 'w+') as f:
        for idx, ind in pedigree.writeOrder():
            matrix = genoProbFunc(ind)
            f.write('\n')
            for i in range(matrix.shape[0]) :
                f.write(ind.idx + ' ' + ' '.join(map("{:.4f}".format, matrix[i,:])) + '\n')

# def reverse_individual(ind):
#     new_ind = ImputationIndividual.AlphaImputeIndividual(ind.idx, ind.idn)
#     new_ind.genotypes = np.ascontiguousarray(np.flip(ind.genotypes))

#     new_ind.setupIndividual()
#     Imputation.ind_align(new_ind)
#     return(new_ind)

# def add_backward_info(ind, rev_ind):
#     ind.phasing_view.backward[:,:] = np.flip(rev_ind.phasing_view.forward, axis = 1) # Flip along loci.

def collapse_and_call(ind, rev_ind):

    ind.phasing_view.backward = np.flip(rev_ind.phasing_view.forward, axis = 1) # Flip along loci.
    ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)

    combined = np.log(ind.phasing_view.backward) + np.log(ind.phasing_view.forward) + np.log(ind.phasing_view.penetrance)

    Heuristic_Peeling.exp_2D_norm(combined, combined)
    ind.phasing_view.called_genotypes = call_genotypes(combined)


def call_genotypes(matrix):
    matrixCollapsedHets = np.array([matrix[0,:], matrix[1,:] + matrix[2,:], matrix[3,:]], dtype=np.float32)
    calledGenotypes = np.argmax(matrixCollapsedHets, axis = 0)
    return calledGenotypes.astype(np.int8)

@profile
def main():
    
    # Set up arguments and pedigree
    args = getArgs()
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    
    startTime = datetime.datetime.now()
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)
    print("Readin", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    # Fill in haplotypes and phase
    setupImputation(pedigree)

    if args.phase:
        hd_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9)  > .6]
        cycles = [args.n_phasing_particles for i in range(args.n_phasing_cycles)]

        rev_individuals = [ind.reverse_individual() for ind in hd_individuals]
        ParticlePhasing.create_library_and_phase(rev_individuals, cycles, args)     

        for ind in hd_individuals:
            ind.add_backward_info()

        ParticlePhasing.create_library_and_phase(hd_individuals, cycles, args)     

        if args.popimpute:
            ld_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9) < 1]

            flipped_dict = dict()
            reversed_ld = [ind.reverse_individual() for ind in ld_individuals]

            library = ParticlePhasing.get_reference_library(hd_individuals, setup = False, reverse = True)
            ParticleImputation.impute_individuals_with_bw_library(reversed_ld, library, n_samples = args.n_imputation_particles)

            for ind in ld_individuals:
                ind.add_backward_info()


            print(len(ld_individuals), "Sent to imputation")
            library = ParticlePhasing.get_reference_library(hd_individuals, setup = False)
            ParticleImputation.impute_individuals_with_bw_library(ld_individuals, library, n_samples = args.n_imputation_particles)


    # Run family based phasing.
    if args.pedimpute:
        Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = .1)

    # Write out results
    startTime = datetime.datetime.now()

    writeOutResults(pedigree, args)
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()


if __name__ == "__main__":
    main()