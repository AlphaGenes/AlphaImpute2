from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import ParticlePhasing
from .Imputation import ParticleImputation
from .Imputation import Heuristic_Peeling
from .Imputation import ImputationIndividual
from .Imputation import Imputation
from .Imputation import ArrayClustering

from .tinyhouse.Utils import time_func

# try:
from .Imputation import version
version_verion = version.version
version_commit = version.commit
version_date = version.date


# except:
#     version_verion = None
#     version_commit = None
#     version_date = None

import argparse
import numpy as np

try:
    profile
except:
    def profile(x): 
        return x


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
    core_impute_parser.add_argument('-length', default=1, required=False, type=float, help='Estimated map length for pedigree and population imputation in Morgans. Default: 1 (100cM).')


    pedigree_parser = parser.add_argument_group("Pedigree imputation options")

    pedigree_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles. Default: 4')
    pedigree_parser.add_argument('-final_peeling_threshold',default=0.1, required=False, type=float, help='Genotype calling threshold for final round of peeling. Default: 0.1 (best guess genotypes).')
    pedigree_parser.add_argument('-cutoff',default=.95, required=False, type=float, help = argparse.SUPPRESS) #help='Genotype calling threshold. Default: 0.95.')

    population_parser = parser.add_argument_group("Population imputation options")

    population_parser.add_argument('-n_phasing_cycles',default=5, required=False, type=int, help='Number of phasing cycles. Default: 5')
    population_parser.add_argument('-n_phasing_particles',default=40, required=False, type=int, help='Number of phasing particles. Defualt: 40.')
    population_parser.add_argument('-n_imputation_particles',default=100, required=False, type=int, help='Number of imputation particles. Defualt: 100.')

    population_parser.add_argument('-hd_threshold',default=0.95, required=False, type=float, help='Percentage of non-missing markers for an individual be classified as high-density when building the haplotype library. Default: 0.95.')
    population_parser.add_argument('-min_chip',default=100, required=False, type=float, help='Minimum number of individuals on an inferred low-density chip for it to be considered a low-density chip. Default: 0.05')
    population_parser.add_argument('-phasing_loci_inclusion_threshold', default=0.9, required=False, type=float, help='Percentage of non-missing markers per loci for it to be included on a chip for imputation. Default: 0.9.')
    
    population_parser.add_argument('-imputation_length_modifier', default=1, required=False, type=float, help='Increases the effective map length of the chip for population imputation by this amount. Default: 1.')
    population_parser.add_argument('-phasing_length_modifier', default=5, required=False, type=float, help='Increases the effective map length of the chip for Phasing imputation by this amount. Default: 5.')
    population_parser.add_argument('-phasing_consensus_window_size', default=50, required=False, type=int, help='Number of markers used to evaluate haplotypes when creating a consensus haplotype. Default: 50.')
    

    integrated_parser = parser.add_argument_group("Joint imputation options") 
    integrated_parser.add_argument('-chip_threshold',default=0.95, required=False, type=float, help='Proportion more high density markers parents need to be used over population imputation. Default: 0.95')
    integrated_parser.add_argument('-final_peeling_threshold_for_phasing',default=0.9, required=False, type=float, help='Genotype calling threshold for first round of peeling before phasing. This value should be conservative.. Default: 0.9.')
    integrated_parser.add_argument('-lazy_phasing', action='store_true', required=False, help = argparse.SUPPRESS) #help='Flag to use pedigree-phased HD individuals as the haplotype reference library. This option decreases runtime at the cost of accuracy')
    integrated_parser.add_argument('-prioritze_individual', action='store_true', required=False, help = argparse.SUPPRESS) # help='Flag to prioritze pedigree imputation for individuals at the same genotyping density as their parents.')
    
    return InputOutput.parseArgs("AlphaImpute", parser)


def writeGenoProbs(pedigree, genoProbFunc, outputFile):
    # Function to write out the penetrance/anterior/posterior terms for individuals.
    # Used for debuging heuristic peeling
    with open(outputFile, 'w+') as f:
        for idx, ind in pedigree.writeOrder():
            matrix = genoProbFunc(ind)
            f.write('\n')
            for i in range(matrix.shape[0]) :
                f.write(ind.idx + ' ' + ' '.join(map("{:.4f}".format, matrix[i,:])) + '\n')


def create_haplotype_library(hd_individuals, args):

    # Adds an initial 1-particle phasing cycle to jump start the process.
    cycles = [1] + [args.n_phasing_particles for i in range(args.n_phasing_cycles)]
    ParticlePhasing.run_phasing(hd_individuals, cycles, args)

    for individual in hd_individuals:
        # Current haplotypes are the haplotypes from the last round of phasing.
        individual.clear_phasing_view(keep_current_haplotypes = True)


    return hd_individuals


def run_population_imputation(ld_individuals, args, haplotype_library, arrays):
    # Split up the population based on existing LD panels, and then impute each panel.
    ld_arrays = ArrayClustering.create_array_subset(ld_individuals, arrays, min_markers = 0, min_individuals = 0)

    for i, chip in enumerate(ld_arrays):
        print("")
        print(f"Imputing chip {i+1} of {len(ld_arrays)}")
        ParticleImputation.impute_individuals_on_chip(chip.individuals, args, haplotype_library)


def run_population_only(pedigree, arrays, args):
    # One of three ways to run the program -- uses only population imputation. For use in populations with unknown pedigree
    # 1. Creates a set of HD individuals based on args.hd_threshold.
    # 2. Runs phasing on HD individuals 
    # 3. Run imputation on LD individuals

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


def run_combined(pedigree, arrays, args):
    # One of three ways to run the program -- for general use for populations with known/unknown pedigree
    # 1. Runs pedigree imputation with a conservative cutoff.
    # 1a. Runs peeling using: Heuristic_Peeling.run_integrated_peeling
    # 1b. Masks individuals back down to their original genotyping array (this is important to help with reducing missing markers for phasing)
    # 1c. Splits the population out into HD individuals, ld for population imputation, and ld for pedigree imputation 
    # 2. Runs phasing on HD individuals 
    # 3. Run imputation on LD individuals for population imputation
    # 4. Lesions the pedigree to block parent information for pseudofounders.
    # 5. Run final round of peeling.
    
    print_title("Population and Pedigree Imputation")

    for ind in pedigree:
        if np.mean(ind.genotypes != 9) > args.hd_threshold :
            ind.original_hd = True
        else:
            ind.original_hd = False


    set_decision_rule(pedigree, args)

    print_title("Initial Pedigree Imputation")
    print(f"Number of peeling cycles: {args.cycles}")
    print(f"Final cutoff before population imputation: {args.final_peeling_threshold_for_phasing}")

    hd_individuals, ld_for_pop_imputation, ld_for_ped_imputation = Heuristic_Peeling.run_integrated_peeling(pedigree, args, final_cutoff = args.final_peeling_threshold_for_phasing, arrays = arrays)


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
        ind.mask_parents() # Lesions the pedigree.
    pedigree.reset_families()

    # Run final round of heuristic peeling, peeling down.
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = args.final_peeling_threshold)

    for ind in population_targets:
        ind.unmask_parents()
    pedigree.reset_families()


def set_decision_rule(pedigree, args):
    for ind in pedigree:
        if args.prioritze_individual:
            ind.marker_score_decision_rule = ind.marker_score_decision_rule_prioritize_individual
        
        else :
            ind.marker_score_decision_rule = ind.marker_score_decision_rule_prioritize_parents


def run_pedigree_only(pedigree, args):
    # One of three ways to run the program -- for use in populations with pedigree information where fast imputation needs to be done.
    # Will lead to low accuracy on founders of the population
    # 1. Runs pedigree imputation.

    print_title("Pedigree Imputation Only")
    print(f"Number of peeling cycles: {args.cycles}")
    print(f"Final cutoff: {args.final_peeling_threshold}")
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = args.final_peeling_threshold)


def run_cluster_only(pedigree, args):
    # Clusters individuals based on their array and writes out the results.

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
            pedigree.writePhase(args.out + ".haplotypes")


@time_func("Full Program Run")
def main():
    InputOutput.print_boilerplate("AlphaImpute2", version_verion, version_commit, version_date)
    args = getArgs()

    InputOutput.setNumbaSeeds(12345)
    pedigree = Pedigree.Pedigree(constructor = ImputationIndividual.AlphaImputeIndividual) 
    read_in_data(pedigree, args)
    for ind in pedigree :
        ind.map_length = args.length
        ind.setupIndividual()
        Imputation.ind_align(ind)

    # First check if clustering only.
    if args.cluster_only:
        run_cluster_only(pedigree, args)
        exit()


    # Set up run mode. Default: Joint. Options Joint, Ped_only, Pop_only
    if args.ped_only:
        run_mode = "ped_only"
    elif args.pop_only:
        run_mode = "pop_only"
    else:
        run_mode = "joint"

    if args.ped_only and args.pop_only:
        print("Given arguments -ped_only and -pop_only. Running in joint mode.")
        run_mode = "joint"
    
    if run_mode == "ped_only":
        run_pedigree_only(pedigree, args)

    if run_mode == "pop_only" or run_mode == "joint":
        print_title("Array Clustering")
        arrays = ArrayClustering.cluster_individuals_by_array([ind for ind in pedigree], args.min_chip)
        print(arrays)

        if run_mode == "pop_only":
            run_population_only(pedigree, arrays, args)
        if run_mode == "joint":
            run_combined(pedigree, arrays, args)


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