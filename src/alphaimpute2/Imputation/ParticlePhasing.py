import numpy as np

import concurrent.futures

from numba import jit


from itertools import repeat

from . import BurrowsWheelerLibrary
from . import PhasingObjects

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

# try:
if "profile" not in locals():

    def profile(x):
        return x


# @time_func("Creating BW library")
@profile
def get_reference_library(individuals, individual_exclusion=False, reverse=False):
    # Construct a library, and add individuals to it.
    # individual_exclusion adds a flag to record who's haplotype is in the library, so that the haplotype can be removed when phasing that individual.
    # setup: Determins whether the BW library is set up, or if just a base library is created. This can be useful if the library needs to be sub-setted before being used.
    # Reverse: Determines if the library should be made up of the reverse haplotypes -- this is used for the backward passes.

    haplotype_library = BurrowsWheelerLibrary.BurrowsWheelerLibrary()

    for ind in individuals:
        for hap in ind.current_haplotypes:
            # Unless set to something else, ind.current_haplotypes tracks ind.haplotypes.
            if reverse:
                new_hap = np.ascontiguousarray(np.flip(hap))
            else:
                new_hap = np.ascontiguousarray(hap.copy())

            if individual_exclusion:
                haplotype_library.append(new_hap, ind)
            else:
                haplotype_library.append(new_hap)

    # Fills in missing data, runs the BW algorithm on the haplotypes, and sets exclusions.
    return haplotype_library


def run_phasing(individuals, cycles, args):
    # Runs phasing cycles. Note: All backward cycles are run before running the forward cycles.
    # I tried running it with running forward/backward/forward/backward,
    # and the contamination of the forward cycles in the backward pass substantially decreased accuracy.

    print("")
    print("Backwards phasing cycles.")
    rev_individuals = setup_reverse_individuals(individuals)
    create_library_and_phase(rev_individuals, cycles, args)
    integrate_reverse_individuals(individuals)
    rev_individuals = None

    print("")
    print("Forwards phasing cycles")
    create_library_and_phase(individuals, cycles, args)


def setup_reverse_individuals(individuals):
    rev_individuals = [ind.reverse_individual() for ind in individuals]
    # Run reverse pass
    for rev_ind in rev_individuals:
        rev_ind.setPhasingView()
    return rev_individuals


def integrate_reverse_individuals(individuals):
    for ind in individuals:
        ind.add_backward_info()
        ind.clear_reverse_view()
        ind.setPhasingView()


@time_func("Total phasing")
@profile
def create_library_and_phase(individuals, cycles, args):
    # This function creates a haplotype library and phases individuals using the haplotype library.
    # Set haplotypes on the last sample.

    for i in range(len(cycles) - 1):
        phase_round(individuals, set_haplotypes=False, n_samples=cycles[i], args=args)

    # Run last round of phasing.
    phase_round(individuals, set_haplotypes=True, n_samples=cycles[-1], args=args)


@time_func("Phasing round")
@profile
def phase_round(individuals, set_haplotypes=False, n_samples=40, args=None):
    bw_library = get_reference_library(individuals, individual_exclusion=True)
    bw_library.setup_library(create_reverse_library=True, create_a=False)

    if InputOutput.args.maxthreads <= 1:
        for individual in individuals:
            phase_individual(
                individual,
                bw_library,
                set_haplotypes=set_haplotypes,
                n_samples=n_samples,
                map_length=args.length * args.phasing_length_modifier,
            )

    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=InputOutput.args.maxthreads
        ) as executor:
            executor.map(
                phase_individual,
                individuals,
                repeat(bw_library),
                repeat(set_haplotypes),
                repeat(n_samples),
                repeat(args.length * args.phasing_length_modifier),
            )


def phase_individual(
    individual, haplotype_library, set_haplotypes, n_samples, map_length
):
    phase(
        individual,
        haplotype_library,
        set_haplotypes=set_haplotypes,
        imputation=False,
        n_samples=n_samples,
        map_length=map_length,
    )


def get_random_values(individual, library, n_samples):
    nHaps, n_loci = library.zeroOccNext.shape
    random_values = individual.random_generator.random(size=(n_samples, n_loci))
    return random_values


def phase(
    individual, haplotype_library, set_haplotypes, imputation, n_samples, map_length
):
    # Random values ensures that phasing is consistent per individual
    random_values = get_random_values(individual, haplotype_library.library, n_samples)
    individual.phasing_view.setup_penetrance()
    phase_jit(
        individual.phasing_view,
        haplotype_library.library,
        set_haplotypes,
        imputation,
        n_samples,
        random_values,
        map_length,
        InputOutput.args.phasing_consensus_window_size,
    )
    individual.phasing_view.clear_penetrance()


@jit(nopython=True, nogil=True)
def phase_jit(
    ind,
    haplotype_library,
    set_haplotypes,
    imputation,
    n_samples,
    random_values,
    map_length,
    phasing_consensus_window_size,
):
    # Phases a specific individual.
    # Set_haplotypes determines whether or not to actually set the haplotypes of an individual based on the underlying samples.
    # Set_haploypes also determines whether forward_geno_probs gets calculated.

    # FLAG: error_rate is hard coded.

    nLoci = haplotype_library.nLoci
    rate = map_length / nLoci

    if imputation:
        calculate_forward_estimates = True
        track_hap_info = True
    else:
        calculate_forward_estimates = set_haplotypes
        track_hap_info = False

    error_rate = 0.01

    sample_container = PhasingObjects.PhasingSampleContainer(haplotype_library, ind)
    for i in range(n_samples):
        # This is the main imputation step.
        sample_container.add_sample(
            rate,
            error_rate,
            calculate_forward_estimates,
            track_hap_info,
            random_values[i, :],
        )

    if imputation:
        # For imputation phasing is only run on a subset of loci.
        # This step extends the phase information to all of the loci.
        converted_samples = [
            expand_sample(ind, sample, haplotype_library)
            for sample in sample_container.samples
        ]
        extended_sample_container = PhasingObjects.PhasingSampleContainer(
            haplotype_library, ind
        )
        extended_sample_container.samples = converted_samples
        pat_hap, mat_hap = extended_sample_container.get_consensus(
            phasing_consensus_window_size
        )

    else:
        pat_hap, mat_hap = sample_container.get_consensus(phasing_consensus_window_size)

    if not imputation and set_haplotypes:
        if ind.population_imputation_target:
            # If phasing, and individual is a target for imputation, set their haplotypes.
            add_haplotypes_to_ind(ind, pat_hap, mat_hap)

        ind.backward[:, :] = 0
        for sample in sample_container.samples:
            ind.backward += (
                sample.forward.forward_geno_probs
            )  # We're really just averaging over particles.

    if imputation:
        add_haplotypes_to_ind(ind, pat_hap, mat_hap)
        backward = (
            ind.backward
        )  # Not sure why we need to set a secondary variable here, but it turns out we do, otherwise ind.backward doesn't update correctly.

        # Only update loci that were considered as part of the haplotype library.
        for index in haplotype_library.loci:
            for j in range(4):
                backward[j, index] = 0.0

            for sample in sample_container.samples:
                for j in range(4):
                    backward[j, index] += sample.forward.forward_geno_probs[
                        j, index
                    ]  # We're really just averaging over particles.

    # Always set current_haplotype after the last round of phasing.
    ind.current_haplotypes[0][:] = pat_hap
    ind.current_haplotypes[1][:] = mat_hap


@jit(nopython=True, nogil=True)
def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
    # Sets all loci to the new values (this will call missing loci as well.
    ind.haplotypes[0][:] = pat_hap[:]
    ind.haplotypes[1][:] = mat_hap[:]
    ind.genotypes[:] = pat_hap[:] + mat_hap[:]


@jit(nopython=True, nogil=True)
def expand_sample(ind, sample, bw_library):
    nLoci = len(ind.genotypes)

    pat_hap = np.full(nLoci, 9, dtype=np.int8)
    mat_hap = np.full(nLoci, 9, dtype=np.int8)

    hap_info = sample.hap_info

    # Fill in the missing loci for phasing.
    for i in range(len(hap_info.pat_ranges)):
        range_object = hap_info.pat_ranges[i]
        # Get the start/stop index for the haplotype in terms of the full (global) set of markers
        global_start, global_stop = hap_info.get_global_bounds(i, 0)

        # Expand out the haplotype to the full range
        set_hap_from_range(range_object, pat_hap, global_start, global_stop, bw_library)

    for i in range(len(hap_info.mat_ranges)):
        range_object = hap_info.mat_ranges[i]
        global_start, global_stop = hap_info.get_global_bounds(i, 1)

        set_hap_from_range(range_object, mat_hap, global_start, global_stop, bw_library)

    new_sample = PhasingObjects.PhasingSample(sample.rec_rate, sample.error_rate)
    new_sample.haplotypes = (pat_hap, mat_hap)
    new_sample.genotypes = pat_hap + mat_hap

    # Sets the recombination score for the expanded sample.
    # Recombination scores are just applied to the corresponding loci in the global set of markers.
    # Missing markers on the chip, have a recombination score of 0.
    new_sample.rec = expand_rec_tracking(sample.rec, bw_library)

    return new_sample


@jit(nopython=True, nogil=True)
def expand_rec_tracking(partial_rec, bw_library):
    new_rec = np.full(bw_library.full_nLoci, 0, dtype=np.float32)
    for i in range(len(partial_rec)):
        true_index = bw_library.get_true_index(i)
        new_rec[true_index] = partial_rec[i]

    return new_rec


@jit(nopython=True, nogil=True)
def set_hap_from_range(range_object, hap, global_start, global_stop, bw_library):
    encoding_index = range_object.encoding_index

    # Select the middle haplotype.
    bw_index = int((range_object.hap_range[0] + range_object.hap_range[1] - 1) / 2)
    haplotype_index = bw_library.a[bw_index, encoding_index]
    hap[global_start:global_stop] = bw_library.full_haps[
        haplotype_index, global_start:global_stop
    ]
