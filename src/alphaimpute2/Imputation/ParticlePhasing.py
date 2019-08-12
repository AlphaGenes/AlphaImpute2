import numba
import numpy as np
import random
import concurrent.futures

from numba import njit, jit, jitclass
from collections import OrderedDict

from itertools import repeat

from . import BurrowsWheelerLibrary
from . import Imputation
from . import ImputationIndividual
from . import PhasingObjects

from . import tmp_merging

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

try:
    profile
except:
    def profile(x): 
        return x


@time_func("Total phasing")
@profile
def create_library_and_phase(individuals, pedigree, args, final_phase = True) :
    # This function creates a haplotype library and phases individuals using the haplotype library.

    for ind in individuals:
        # We never call genotypes so can do this once.
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)

    for rep in range(5):
        phase_round(individuals, individual_exclusion = True, set_haplotypes = False)
    
    # Second round of genotype calling
    for ind in individuals:
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)
       
    phase_round(individuals, individual_exclusion = True, set_haplotypes = True)


@time_func("Phasing round")
@profile
def phase_round(individuals, individual_exclusion = False, set_haplotypes = False):
    # In a given round we create a haplotype reference library, and phase individuals using it.
    bwLibrary = get_reference_library(individuals, individual_exclusion)
    phase_individuals_with_bw_library(individuals, bwLibrary, set_haplotypes = set_haplotypes)

@time_func("Creating BW library")
@profile
def get_reference_library(individuals, individual_exclusion = False, setup = True):
    # Construct a library, and add individuals to it.
    # If we are worried about an individual's haplotype being included in the reference library (i.e. because we are about to phase that individual)
    # Then use the individual_exclusion flag to make sure they don't use their own haplotype.

    haplotype_library = BurrowsWheelerLibrary.BurrowsWheelerLibrary()
    
    for ind in individuals:
        for hap in ind.phasing_view.current_haplotypes:
            # Unless set to something else, ind.current_haplotypes tracks ind.haplotypes.
            if individual_exclusion:
                haplotype_library.append(hap.copy(), ind)
            else:
                haplotype_library.append(hap.copy())

    # Fills in missing data, runs the BW algorithm on the haplotypes, and sets exclusions.
    if setup:
        haplotype_library.setup_library()
    return haplotype_library



def phase_individuals_with_bw_library(individuals, bwLibrary, set_haplotypes):
    # This tiny function is split out since we may want to run it from something else.
    chunksize = 100
    jit_individuals = [ind.phasing_view for ind in individuals]

    if InputOutput.args.maxthreads <= 1 or len(individuals) < chunksize:
        phase_group(jit_individuals, bwLibrary.library, set_haplotypes = set_haplotypes)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=InputOutput.args.maxthreads) as executor:
            groups = split_individuals_into_groups(jit_individuals, chunksize)
            executor.map(phase_group, groups, repeat(bwLibrary.library), repeat(set_haplotypes))

def split_individuals_into_groups(individuals, chunksize):
    # This seems to handle the end of lists okay.
    groups = [individuals[start:(start+chunksize)] for start in range(0, len(individuals), chunksize)]
    return groups

# @jit(nopython=True, nogil=True) 
def phase_group(individuals, haplotype_library, set_haplotypes):
    for ind in individuals:
        phase(ind, haplotype_library, set_haplotypes = set_haplotypes)


# @jit(nopython=True, nogil=True) 
def phase(ind, haplotype_library, set_haplotypes = False) :
    nLoci = len(ind.genotypes)
    rate = 5/nLoci


    if set_haplotypes:
        n_samples = 40
        error_rate = 0.01
    else:
        error_rate = 0.01
        n_samples = 40

    samples = PhasingObjects.PhasingSampleContainer(haplotype_library, ind)
    for i in range(n_samples):
        samples.add_sample(rate, error_rate)


    if n_samples == 1:
        pat_hap, mat_hap = samples.get_consensus(50)
    else:
        # samples = tmp_merging.backwards_sampling(samples)
        pat_hap, mat_hap = samples.get_consensus(50)

    if set_haplotypes:
        add_haplotypes_to_ind(ind, pat_hap, mat_hap)
    else:
        ind.current_haplotypes[0][:] = pat_hap
        ind.current_haplotypes[1][:] = mat_hap
    

@jit(nopython=True, nogil=True) 
def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
    nLoci = len(pat_hap)
    for i in range(nLoci):
        # This is weird becuase it was designed to have an option to not fill in missing genotypes. 
        # It looks like filling in missing genotypes is okay though.
        ind.haplotypes[0][i] = pat_hap[i]
        ind.haplotypes[1][i] = mat_hap[i]
        ind.genotypes[i] = pat_hap[i] + mat_hap[i]

