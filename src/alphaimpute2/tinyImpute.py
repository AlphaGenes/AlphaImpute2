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
    core_impute_parser.add_argument('-ped_finish', action='store_true', required=False, help='Flag to run the pedigree imputation after population imputation.')
   
    core_impute_parser.add_argument('-cluster_only', action='store_true', required=False, help='Flag to just cluster individuals into marker arrays and write out results.')

    core_impute_parser.add_argument('-cutoff',default=.95, required=False, type=float, help='Genotype calling threshold.')
    core_impute_parser.add_argument('-cycles',default=4, required=False, type=int, help='Number of peeling cycles.')
    core_impute_parser.add_argument('-final_peeling_threshold',default=0.1, required=False, type=float, help='Genotype calling threshold for final round of peeling. Default: 0.1 (best guess genotypes).')
    core_impute_parser.add_argument('-final_peeling_threshold_for_phasing',default=0.98, required=False, type=float, help='Genotype calling threshold for final round of peeling when phasing is run. Default: 0.98.')


    core_impute_parser.add_argument('-n_phasing_particles',default=40, required=False, type=int, help='Number of phasing particles. Defualt: 40.')
    core_impute_parser.add_argument('-n_phasing_cycles',default=5, required=False, type=int, help='Number of phasing cycles. Default: 4')

    core_impute_parser.add_argument('-n_imputation_particles',default=100, required=False, type=int, help='Number of imputation particles. Defualt: 100.')

    core_impute_parser.add_argument('-hd_threshold',default=0.9, required=False, type=float, help='Threshold for high density individuals when building the haplotype library. Default: 0.8.')
    core_impute_parser.add_argument('-min_chip',default=100, required=False, type=float, help='Minimum number of individuals on an inferred low-density chip for it to be considered a low-density chip. Default: 0.05')
    
    prephase_parser = parser.add_argument_group("Prephase options")
    prephase_parser.add_argument('-allow_prephased_bypass', action='store_true', required=False, help='Allow the algorithm to use pedigree phased haplotypes to create a reference library.')

    prephase_parser.add_argument('-prephased_threshold',default=5000, required=False, type=int, help='Number of individuals required to be fully phased before using the pre-phase bypass. Default: 5000.')


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

def create_haplotype_library(hd_individuals, args):

    pre_phase_ran = False

    if args.allow_prephased_bypass:

        # Try to use prephased haplotypes to do the phasing -- if so, run two rounds of phasing and then a round of imputation.
        # Need to tune parameters here...

        prephased_hd_individuals = [ind for ind in hd_individuals if ind.percent_phased > .95]
        unphased_hd_individuals = [ind for ind in hd_individuals if ind.percent_phased <= .95]

        if len(prephased_hd_individuals) > args.prephased_threshold :
            pre_phase_ran = True
            
            print("Phasing", len(prephased_hd_individuals), "pre-phased HD individuals")
            cycles = [args.n_phasing_particles]*2
            run_phasing(prephased_hd_individuals, cycles, args)


            print("Phasing remaining HD individuals.")
            impute_individuals_on_chip(unphased_hd_individuals, args, prephased_hd_individuals)

    if not pre_phase_ran :

        print("Phasing", len(hd_individuals), "HD individuals")
        cycles = [1] + [args.n_phasing_particles for i in range(args.n_phasing_cycles)]
        run_phasing(individuals, cycle, args)

    return hd_individuals


def run_phasing(individuals, cycles, args):
    print("Running backward passes")
    rev_individuals = setup_reverse_individuals(individuals)
    ParticlePhasing.create_library_and_phase(rev_individuals, cycles, args)     
    integrate_reverse_individuals(individuals)

    print("Running forward passes")
    ParticlePhasing.create_library_and_phase(individuals, cycles, args)     



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


def run_population_imputation(ld_individuals, args, haplotype_library, arrays):

    print("Splitting individuals into different marker densities")

    ArrayClustering.update_arrays(arrays)

    ld_arrays = ArrayClustering.create_array_subset(ld_individuals, arrays, min_markers = 0, min_individuals = 0)

    ld_arrays.write_out_arrays(args.out + "post_ped.arrays")


    for chip in ld_arrays:
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


def run_pedigree_only(pedigree, args):

    final_cutoff = args.final_peeling_threshold
    Heuristic_Peeling.runHeuristicPeeling(pedigree, args, final_cutoff = final_cutoff)

def run_population_only(pedigree, arrays, args):

    hd_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9)  > args.hd_threshold]

    haplotype_library = create_haplotype_library(hd_individuals, args)

    ld_individuals = [ind for ind in pedigree if np.mean(ind.genotypes != 9) <= args.hd_threshold]
    run_population_imputation(ld_for_pop_imputation, args, haplotype_library, arrays)

def run_joint(pedigree, arrays, args):

    final_cutoff = args.final_peeling_threshold_for_phasing
    
    hd_individuals, ld_for_pop_imputation, ld_for_ped_imputation = Heuristic_Peeling.run_integrated_peeling(pedigree, args, final_cutoff = final_cutoff)

    haplotype_library = create_haplotype_library(hd_individuals, args)

    print("Total: ", len(ld_individuals), "Post Filter: ", len(ld_for_pop_imputation))

    run_population_imputation(ld_for_pop_imputation, args, haplotype_library, arrays)

    # Run final round of heuristic peeling, peeling down.
    Heuristic_Peeling.run_integrated_peel_down(pedigree, ld_for_ped_imputation, args, final_cutoff = .1) 




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

    pedigree_only = args.pedimpute and not (args.phase or args.popimpute)
    pop_only = (args.phase or args.popimpute) and not args.pedimpute
    joint = (args.phase or args.popimpute) and args.pedimpute

    if args.cluster_only:
        arrays = ArrayClustering.cluster_individuals_by_array(individuals, args.min_chip)
        arrays.write_out_arrays(args.out + ".arrays")
        exit()

    elif pedigree_only:
        run_pedigree_only(pedigree, args)

    elif pop_only or joint:
        arrays = ArrayClustering.cluster_individuals_by_array(individuals, args.min_chip)

        if pop_only:
            run_population_only(pedigree, arrays, args)
        if joint:
            run_joint(pedigree, arrays, args)

    # Write out results
    startTime = datetime.datetime.now()
    writeOutResults(pedigree, args)
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()


if __name__ == "__main__":
    main()