from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import ParticlePhasing
from .Imputation import ParticleImputation
from .Imputation import Heuristic_Peeling
from .Imputation import ImputationIndividual
from .Imputation import Imputation
from .Imputation import ArrayClustering

from .tinyhouse.Utils import time_func

import datetime
import argparse
import numpy as np
from numba import njit

try:
    profile
except:
    def profile(x): 
        return x

def setupImputation(pedigree):
    for ind in pedigree :
        ind.setupIndividual()
        Imputation.ind_align(ind)



def getArgs() :
    parser = argparse.ArgumentParser(description='')
    core_parser = parser.add_argument_group("Core arguments")
    core_parser.add_argument('-out', required=True, type=str, help='The output file prefix.')
    InputOutput.addInputFileParser(parser)

    probability_parser = parser.add_argument_group("Probability options")
    InputOutput.add_arguments_from_dictionary(probability_parser, InputOutput.get_probability_options(), ["error"])

    core_impute_parser = parser.add_argument_group("Imputation options")
    core_impute_parser.add_argument('-maxthreads',default=1, required=False, type=int, help='Number of threads to use. Default: 1.')
    core_impute_parser.add_argument('-binaryoutput', action='store_true', required=False, help='Flag to write out the genotypes as a binary plink output.')
    core_impute_parser.add_argument('-phase_output', action='store_true', required=False, help='Flag to write out the phase information.')
    
    core_impute_parser.add_argument('-pop_only', action='store_true', required=False, help='Flag to run the population based imputation algorithm only.')
    core_impute_parser.add_argument('-ped_only', action='store_true', required=False, help='Flag to run the pedigree based imputation algorithm only.')  
    core_impute_parser.add_argument('-cluster_only', action='store_true', required=False, help='Flag to just cluster individuals into marker arrays and write out results.')


    pedigree_parser = parser.add_argument_group("Pedigree imputation options")

    pedigree_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles. Default: 4')
    pedigree_parser.add_argument('-final_peeling_threshold',default=0.1, required=False, type=float, help='Genotype calling threshold for final round of peeling. Default: 0.1 (best guess genotypes).')
    pedigree_parser.add_argument('-cutoff',default=.95, required=False, type=float, help = argparse.SUPPRESS) #help='Genotype calling threshold. Default: 0.95.')

    population_parser = parser.add_argument_group("Population imputation options")

    population_parser.add_argument('-n_phasing_cycles',default=5, required=False, type=int, help='Number of phasing cycles. Default: 5')
    population_parser.add_argument('-n_phasing_particles',default=40, required=False, type=int, help='Number of phasing particles. Defualt: 40.')
    population_parser.add_argument('-n_imputation_particles',default=100, required=False, type=int, help='Number of imputation particles. Defualt: 100.')

    population_parser.add_argument('-hd_threshold',default=0.95, required=False, type=float, help='Percentage of non-missing markers to be classified as high-density when building the haplotype library. Default: 0.95.')
    population_parser.add_argument('-min_chip',default=100, required=False, type=float, help='Minimum number of individuals on an inferred low-density chip for it to be considered a low-density chip. Default: 0.05')
    

    integrated_parser = parser.add_argument_group("Joint imputation options") 
    integrated_parser.add_argument('-chip_threshold',default=0.95, required=False, type=float, help='Proportion more high density markers parents need to be used over population imputation. Default: 0.95')
    integrated_parser.add_argument('-final_peeling_threshold_for_phasing',default=0.9, required=False, type=float, help='Genotype calling threshold for first round of peeling before phasing. This value should be conservative.. Default: 0.9.')
    integrated_parser.add_argument('-integrated_decision_rule',default="individual", required=False, type=str, help='Decision rule to use when determining whether to use population or pedigree imputation. Options: individual, balanced, parents. Default: individual')
    integrated_parser.add_argument('-joint_type',default="pedigree", required=False, type=str, help='Decision rule to use when determining which joint option to use. Options: integrated, pedigree. Default: pedigree')
    integrated_parser.add_argument('-lazy_phasing', action='store_true', required=False, help='Flag to use pedigree-phased HD individuals as the haplotype reference library.')
    integrated_parser.add_argument('-defer_parents', action='store_true', required=False, help='Flag to prioritze pedigree imputation for individuals at the same genotyping density as their parents.')


    # prephase_parser = parser.add_argument_group("Prephase options")
    # prephase_parser.add_argument('-allow_prephased_bypass', action='store_true', required=False, help='Allow the algorithm to use pedigree phased haplotypes to create a reference library.')
    # prephase_parser.add_argument('-prephased_threshold',default=5000, required=False, type=int, help='Number of individuals required to be fully phased before using the pre-phase bypass. Default: 5000.')
    
    return InputOutput.parseArgs("AlphaImpute", parser)


def writeGenoProbs(pedigree, genoProbFunc, outputFile):
    with open(outputFile, 'w+') as f:
        for idx, ind in pedigree.writeOrder():
            matrix = genoProbFunc(ind)
            f.write('\n')
            for i in range(matrix.shape[0]) :
                f.write(ind.idx + ' ' + ' '.join(map("{:.4f}".format, matrix[i,:])) + '\n')


def create_haplotype_library(hd_individuals, args):

    pre_phase_ran = False

    # if False: # args.allow_prephased_bypass:

    #     # Try to use prephased haplotypes to do the phasing -- if so, run two rounds of phasing and then a round of imputation.
    #     # Need to tune parameters here...

    #     prephased_hd_individuals = [ind for ind in hd_individuals if ind.percent_phased > .95]
    #     unphased_hd_individuals = [ind for ind in hd_individuals if ind.percent_phased <= .95]

    #     if len(prephased_hd_individuals) > args.prephased_threshold :
    #         pre_phase_ran = True
            
    #         print("Phasing", len(prephased_hd_individuals), "pre-phased HD individuals")
    #         cycles = [args.n_phasing_particles]*2
    #         run_phasing(prephased_hd_individuals, cycles, args)


    #         print("Phasing remaining HD individuals.")
    #         impute_individuals_on_chip(unphased_hd_individuals, args, prephased_hd_individuals)

    if not pre_phase_ran :
        cycles = [1] + [args.n_phasing_particles for i in range(args.n_phasing_cycles)]
        ParticlePhasing.run_phasing(hd_individuals, cycles, args)

    return hd_individuals


def run_population_imputation(ld_individuals, args, haplotype_library, arrays):
    ld_arrays = ArrayClustering.create_array_subset(ld_individuals, arrays, min_markers = 0, min_individuals = 0)
    for i, chip in enumerate(ld_arrays):
        print("")
        print(f"Imputing chip {i+1} of {len(ld_arrays)}")
        ParticleImputation.impute_individuals_on_chip(chip.individuals, args, haplotype_library)


def run_population_only(pedigree, arrays, args):
    print_title("Population Imputation Only")

    hd_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9)  > args.hd_threshold]
    ld_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9) <= args.hd_threshold]

    print(f"Number of HD individuals: {len(hd_individuals)}")
    print(f"Number of LD individuals: {len(ld_individuals)}")

    print_title("Phasing")
    print(f"Number of phasing cycles: {args.n_phasing_cycles}")
    print(f"Number of phasing particles: {args.n_phasing_particles}")

    haplotype_library = create_haplotype_library(hd_individuals, args)

    print_title("Imputation")
    print(f"Number of imputation particles: {args.n_imputation_particles}")
    run_population_imputation(ld_individuals, args, haplotype_library, arrays)

# def run_joint(pedigree, arrays, args):
#     # Set decision rule
#     set_decision_rule(pedigree, args)

#     # original_hd_individuals = [ind for ind in pop_individuals if np.mean(ind.genotypes != 9) > args.hd_threshold]
#     for ind in pedigree:
#         if np.mean(ind.genotypes != 9) > args.hd_threshold :
#             ind.original_hd = True
#         else:
#             ind.original_hd = False

#     final_cutoff = args.final_peeling_threshold_for_phasing
    
#     hd_individuals, ld_for_pop_imputation, ld_for_ped_imputation = Heuristic_Peeling.run_integrated_peeling(pedigree, args, final_cutoff = final_cutoff)

#     for ind in hd_individuals:
#         ind.population_imputation_target = True

#     pop_individuals = hd_individuals + ld_for_pop_imputation

#     for array in arrays:
#         mask_array(array)

#     hd_individuals = [ind for ind in pop_individuals if np.mean(ind.genotypes != 9) > args.hd_threshold]
#     ld_for_pop_imputation = [ind for ind in pop_individuals if np.mean(ind.genotypes != 9) <= args.hd_threshold]


#     haplotype_library = create_haplotype_library(hd_individuals, args)

#     ld_individuals = ld_for_pop_imputation + ld_for_ped_imputation
#     print("Total: ", len(ld_individuals), "Post Filter: ", len(ld_for_pop_imputation))

#     run_population_imputation(ld_for_pop_imputation, args, haplotype_library, arrays)

#     # Run final round of heuristic peeling, peeling down.
#     Heuristic_Peeling.run_integrated_peel_down(pedigree, ld_for_ped_imputation, args, final_cutoff = .1) 


def run_joint_pedigree_end(pedigree, arrays, args):
    print_title("Population and Pedigree Imputation")

    for ind in pedigree:
        if np.mean(ind.genotypes != 9) > args.hd_threshold :
            ind.original_hd = True
        else:
            ind.original_hd = False


    set_decision_rule(pedigree, args)
    final_cutoff = args.final_peeling_threshold_for_phasing

    print_title("Initial Pedigree Imputation")
    print(f"Number of peeling cycles: {args.cycles}")
    print(f"Final cutoff before population imputation: {final_cutoff}")
    hd_individuals, ld_for_pop_imputation, ld_for_ped_imputation = Heuristic_Peeling.run_integrated_peeling(pedigree, args, final_cutoff = final_cutoff, arrays = arrays)


    print_title("Phasing")
    print(f"Number of HD individuals: {len(hd_individuals)}")
    print(f"Number of phasing cycles: {args.n_phasing_cycles}")
    print(f"Number of phasing particles: {args.n_phasing_particles}")

    if not args.lazy_phasing:
        haplotype_library = create_haplotype_library(hd_individuals, args)

    else:
        haplotype_library = hd_individuals

    ld_individuals = ld_for_pop_imputation + ld_for_ped_imputation
    print_title("Imputation")
    print(f"Number of imputation particles: {args.n_imputation_particles}")
    print(f"Total number of LD individuals: ", len(ld_individuals))
    print(f"Number of LD individuals after filtering: ", len(ld_for_pop_imputation))

    run_population_imputation(ld_for_pop_imputation, args, haplotype_library, arrays)


    print_title("Final Pedigree Imputation")
    print(f"Number of peeling cycles: {args.cycles}")
    print(f"Final cutoff: {args.final_peeling_threshold}")

    population_targets = [ind for ind in pedigree if ind.population_imputation_target]
    for ind in population_targets:
        ind.mask_parents()
    pedigree.reset_families()

    # Run final round of heuristic peeling, peeling down.
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = args.final_peeling_threshold)

    for ind in population_targets:
        ind.unmask_parents()
    pedigree.reset_families()

def mask_array(array):

    mask = array.genotypes
    print(np.sum(mask))
    for ind in array.individuals:
        if not ind.original_hd:
            mask_genotypes(ind.genotypes, mask)
            mask_genotypes(ind.haplotypes[0], mask)
            mask_genotypes(ind.haplotypes[1], mask)

@njit
def mask_genotypes(mat, mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            mat[i] = 9


def set_decision_rule(pedigree, args):
    for ind in pedigree:
        if not args.defer_parents:
            ind.marker_score_decision_rule = ind.marker_score_decision_rule_prioritize_individual
        
        if args.defer_parents:
            ind.marker_score_decision_rule = ind.marker_score_decision_rule_prioritize_parents
        
        # if args.integrated_decision_rule == "balanced":
        #     ind.marker_score_decision_rule = ind.marker_score_decision_rule_prioritize_balanced

def run_pedigree_only(pedigree, args):
    final_cutoff = args.final_peeling_threshold
    print_title("Pedigree Imputation Only")
    print(f"Number of peeling cycles: {args.cycles}")
    print(f"Final cutoff: {final_cutoff}")
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = final_cutoff)


def run_cluster_only(pedigree, args):
    print_title("Array Clustering Only")
    arrays = ArrayClustering.cluster_individuals_by_array([ind for ind in pedigree], args.min_chip)
    print("Final array clusters:")
    print(arrays)
    arrays.write_out_arrays(args.out + ".arrays")

@time_func("Read in")
def read_in_data(pedigree, args):
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)


@time_func("Write out")
def write_out_data(pedigree, args):
    print_title("Writing Out Results")

    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        pedigree.writeGenotypes(args.out + ".genotypes")
        if args.phase_output:
            pedigree.writeGenotypes(args.out + ".haplotypes")


@time_func("Full Program Run")
def main():
    InputOutput.print_boilerplate("AlphaImpute2", "v0.0.1")
    args = getArgs()
    InputOutput.setNumbaSeeds(12345)
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 

    read_in_data(pedigree, args)
    setupImputation(pedigree)

    # If pedigree imputation. Run initial round of pedigree imputation.
    # If no population imputation, run with a low cutoff.

    pedigree_only = args.ped_only
    pop_only = args.pop_only
    joint = not pedigree_only and not pop_only

    if pedigree_only and pop_only:
        print("Given arguments -ped_only and -pop_only. Running in joint mode.")
        joint = True

    if args.cluster_only:
        run_cluster_only(pedigree, args)
        exit()

    elif pedigree_only:
        run_pedigree_only(pedigree, args)

    elif pop_only or joint:
        print_title("Array Clustering")
        arrays = ArrayClustering.cluster_individuals_by_array([ind for ind in pedigree], args.min_chip)
        print(arrays)

        if pop_only:
            run_population_only(pedigree, arrays, args)
        if joint:
            run_joint_pedigree_end(pedigree, arrays, args)


    # Write out results
    write_out_data(pedigree, args)

def print_title(text, center = True):
    print("")
    if center:
        width = 42
        print(f'{text:^{width}}')  # centre aligned
    else:
        width = len(text)
        print(text)
    print('-' * width)


if __name__ == "__main__":
    main()