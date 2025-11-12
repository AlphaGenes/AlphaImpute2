import numpy as np
import concurrent.futures


from numba import jit

from itertools import repeat

from . import ParticlePhasing

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

if not ("profile" in globals()):

    def profile(x):
        return x


@time_func("Chip Imputation")
def impute_individuals_on_chip(ld_individuals, args, haplotype_library):
    # To perform imputation, we set up the haplotype library, and then run phasing in a backwards (reverse) pass and in a forward pass.
    # All individuals sent to imputation are imputed.
    # Most of the actual imputation code is in ParticlePhasing.py and PhasingObjects.py

    forward_loci, reverse_loci = get_non_missing_loci(
        ld_individuals, args.phasing_loci_inclusion_threshold
    )

    if forward_loci is not None:
        print("Number of individuals:", len(ld_individuals))
        print(f"Number of markers: {len(forward_loci)}")

        reverse_library = ParticlePhasing.get_reference_library(
            haplotype_library, reverse=True
        )
        reverse_library.setup_library(loci=reverse_loci, create_a=True)
        multi_threaded_apply(
            backward_impute_individual,
            [ind.reverse_individual() for ind in ld_individuals],
            reverse_library,
            args.n_imputation_particles,
            args.length * args.imputation_length_modifier,
        )
        reverse_library = None

        forward_library = ParticlePhasing.get_reference_library(haplotype_library)
        forward_library.setup_library(loci=forward_loci, create_a=True)
        multi_threaded_apply(
            forward_impute_individual,
            ld_individuals,
            forward_library,
            args.n_imputation_particles,
            args.length * args.imputation_length_modifier,
        )

    else:
        print("Number of individuals:", len(ld_individuals))
        print("Number of markers: 0 - SKIPPED")


def multi_threaded_apply(func, individuals, library, n_particles, map_length):
    if InputOutput.args.maxthreads <= 1:
        for ind in individuals:
            func(ind, library, n_particles, map_length)

    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=InputOutput.args.maxthreads
        ) as executor:
            executor.map(
                func,
                individuals,
                repeat(library),
                repeat(n_particles),
                repeat(map_length),
            )


def backward_impute_individual(reverse_individual, library, n_samples, map_length):
    reverse_individual.setPhasingView()
    ParticlePhasing.phase(
        reverse_individual,
        library,
        set_haplotypes=False,
        imputation=True,
        n_samples=n_samples,
        map_length=map_length,
    )

    individual = reverse_individual.reverse_view
    individual.add_backward_info()
    individual.clear_reverse_view()


def forward_impute_individual(individual, library, n_samples, map_length):
    individual.setPhasingView()
    ParticlePhasing.phase(
        individual,
        library,
        set_haplotypes=False,
        imputation=True,
        n_samples=n_samples,
        map_length=map_length,
    )
    individual.clear_phasing_view(keep_current_haplotypes=False)


def get_non_missing_loci(individuals, threshold):
    # Returns the loci that seem to be present.

    nLoci = len(individuals[0].genotypes)

    # Scores is proportion of missing loci.
    scores = np.full(nLoci, 0, dtype=np.float32)
    for ind in individuals:
        scores += ind.genotypes != 9
    scores /= len(individuals)

    forward_loci = get_loci_pass_threshold(scores, threshold)
    reverse_loci = get_loci_pass_threshold(np.flip(scores), threshold)

    return forward_loci, reverse_loci


@jit(nopython=True, nogil=True)
def get_loci_pass_threshold(scores, threshold):
    # Figure out which loci have a number of missing markers less than threshold.
    loci = None

    for i in range(len(scores)):
        if scores[i] > threshold:
            if loci is None:
                loci = [i]
            else:
                loci += [i]
    return loci
