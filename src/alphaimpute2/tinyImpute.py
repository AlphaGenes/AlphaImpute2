from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import ParticlePhasing
from .Imputation import ParticleImputation
from .Imputation import Heuristic_Peeling
from .Imputation import ImputationIndividual
from .Imputation import Imputation
from .Imputation import ArrayClustering


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

    probability_parser = parser.add_argument_group("Probability options")
    InputOutput.add_arguments_from_dictionary(probability_parser, InputOutput.get_probability_options(), ["error"])

    core_impute_parser = parser.add_argument_group("Impute options")
    core_impute_parser.add_argument('-maxthreads',default=1, required=False, type=int, help='Number of threads to use. Default: 1.')
    core_impute_parser.add_argument('-binaryoutput', action='store_true', required=False, help='Flag to write out the genotypes as a binary plink output.')
    
    core_impute_parser.add_argument('-phase', action='store_true', required=False, help='Flag to run the phasing algorithm.')
    core_impute_parser.add_argument('-popimpute', action='store_true', required=False, help='Flag to run the phasing algorithm.')
    core_impute_parser.add_argument('-pedimpute', action='store_true', required=False, help='Flag to run the pedigree based imputation algorithm.')
   

    core_impute_parser.add_argument('-cutoff',default=.95, required=False, type=float, help='Genotype calling threshold.')
    core_impute_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles.')


    core_impute_parser.add_argument('-n_phasing_particles',default=40, required=False, type=int, help='Number of phasing particles. Defualt: 40.')
    core_impute_parser.add_argument('-n_phasing_cycles',default=5, required=False, type=int, help='Number of phasing cycles. Default: 4')

    core_impute_parser.add_argument('-n_imputation_particles',default=100, required=False, type=int, help='Number of imputation particles. Defualt: 100.')

    core_impute_parser.add_argument('-hd_threshold',default=0.9, required=False, type=float, help='Threshold for high density individuals when building the haplotype library. Default: 0.8.')
    core_impute_parser.add_argument('-min_chip',default=0.05, required=False, type=float, help='Minimum number of individuals on an inferred low-density chip for it to be considered a low-density chip. Default: 0.05')

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


def create_haplotype_library(pedigree, args):

    hd_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9)  > args.hd_threshold]
    
    print("Phasing", len(hd_individuals), "HD individuals")
    print("")
    print("Running backward passes")
    cycles = [args.n_phasing_particles for i in range(args.n_phasing_cycles)]

    rev_individuals = setup_reverse_individuals(hd_individuals)


    ParticlePhasing.create_library_and_phase(rev_individuals, cycles, args)     

    integrate_reverse_individuals(hd_individuals)
    print("")
    print("Running forward passes")

    ParticlePhasing.create_library_and_phase(hd_individuals, cycles, args)     

    return hd_individuals

def setup_reverse_individuals(individuals):

    rev_individuals = [ind.reverse_individual() for ind in individuals]
    # Run reverse pass
    for ind in rev_individuals:
        ind.setPhasingView()

    return rev_individuals

def integrate_reverse_individuals(individuals):

    for ind in individuals:
        ind.add_backward_info()
        ind.clear_reverse_view()
        ind.setPhasingView()


def run_population_imputation(pedigree, args, haplotype_library):

    print("Splitting individuals into different marker densities")

    ld_individuals = [ind for ind in pedigree if (np.mean(ind.genotypes != 9) <= args.hd_threshold and np.mean(ind.genotypes != 9) > 0.01)]
    chips = ArrayClustering.cluster_individuals_by_array(ld_individuals, args.min_chip)

    for chip in chips:
        impute_individuals_on_chip(chip.individuals, args, haplotype_library)


def impute_individuals_on_chip(ld_individuals, args, haplotype_library):

    average_marker_density = np.floor(np.mean([np.sum(ind.genotypes != 9) for ind in ld_individuals]))

    print("Imputing", len(ld_individuals), f"LD individuals genotyped with an average of {average_marker_density} markers")

    flipped_dict = dict()
    reversed_ld = setup_reverse_individuals(ld_individuals)

    print("")
    print("Running backwards passes")

    library = ParticlePhasing.get_reference_library(haplotype_library, setup = False, reverse = True)
    ParticleImputation.impute_individuals_with_bw_library(reversed_ld, library, n_samples = args.n_imputation_particles)

    integrate_reverse_individuals(ld_individuals)

    print("")
    print("Running forward passes")

    library = ParticlePhasing.get_reference_library(haplotype_library, setup = False)
    ParticleImputation.impute_individuals_with_bw_library(ld_individuals, library, n_samples = args.n_imputation_particles)



@profile
def main():
    
    args = getArgs()
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    
    # Read in genotype data, and prepare individuals for imputation.

    startTime = datetime.datetime.now()
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)
    print("Readin", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()
    setupImputation(pedigree)

    individuals = [ind for ind in pedigree]

    # If pedigree imputation. Run initial round of pedigree imputation.
    # If no population imputation, run with a low cutoff.

    if args.pedimpute:

        final_cutoff = 0.1
        if args.phase or args.popimpute:
            final_cutoff = 0.95
        
        Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = .95)

    # If population imputation and phasing, build the haplotype reference panel and impute low density individuals.

    if args.phase or args.popimpute:
        haplotype_library = create_haplotype_library(pedigree, args)

        if args.popimpute:
            run_population_imputation(pedigree, args, haplotype_library)


    # Write out results
    startTime = datetime.datetime.now()
    writeOutResults(pedigree, args)
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()


if __name__ == "__main__":
    main()