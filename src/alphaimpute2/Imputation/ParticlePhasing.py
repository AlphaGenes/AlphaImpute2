import numba
import numpy as np
import random
import concurrent.futures

from numba import njit, jit, prange
from numba.typed import List

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass


from collections import OrderedDict
from itertools import repeat

from . import BurrowsWheelerLibrary
from . import PhasingObjects

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

try:
    profile
except:
    def profile(x): 
        return x


@time_func("Total phasing")
@profile
def create_library_and_phase(individuals, cycles, args) :
    # This function creates a haplotype library and phases individuals using the haplotype library.

    for ind in individuals:
        ind.phasing_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01) # args.error?

    for i in range(len(cycles) - 1):
        phase_round(individuals, individual_exclusion = True, set_haplotypes = False, n_samples = cycles[i])

    # Run last round of phasing.    
    phase_round(individuals, individual_exclusion = True, set_haplotypes = True, n_samples = cycles[-1])


@time_func("Phasing round")
@profile
def phase_round(individuals, individual_exclusion = False, set_haplotypes = False, n_samples = 40):

    bw_library = get_reference_library(individuals, individual_exclusion)
    phase_individuals_with_bw_library(individuals, bw_library, set_haplotypes = set_haplotypes, n_samples = n_samples)


@time_func("Creating BW library")
@profile
def get_reference_library(individuals, individual_exclusion = False, setup = True, reverse = False):
    # Construct a library, and add individuals to it.
    # individual_exclusion adds a flag to record who's haplotype is in the library, so that the haplotype can be removed when phasing that individual.
    # setup: Determins whether the BW library is set up, or if just a base library is created. This can be useful if the library needs to be sub-setted before being used. 
    # Reverse: Determines if the library should be made up of the reverse haplotypes -- this is used for the backward passes.

    haplotype_library = BurrowsWheelerLibrary.BurrowsWheelerLibrary()
    
    for ind in individuals:
        for hap in ind.phasing_view.current_haplotypes:
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
    if setup:
        haplotype_library.setup_library()

    return haplotype_library



def phase_individuals_with_bw_library(individuals, bwLibrary, set_haplotypes, n_samples):
    # Runs a set of individuals with an already-existing BW library and a flag for whether or not to set the haplotypes.
    chunksize = 10
    jit_individuals = List([ind.phasing_view for ind in individuals])
    # for ind in individuals:
    #     jit_individuals
    # jit_individuals = [ind.phasing_view for ind in individuals]

    if InputOutput.args.maxthreads <= 1 or len(individuals) < chunksize:
        phase_group(jit_individuals, bwLibrary.library, set_haplotypes = set_haplotypes, n_samples = n_samples)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=InputOutput.args.maxthreads) as executor:
            groups = split_individuals_into_groups(jit_individuals, chunksize)
            executor.map(phase_group, groups, repeat(bwLibrary.library), repeat(set_haplotypes), repeat(n_samples))

def split_individuals_into_groups(individuals, chunksize):
    # Split out a list of individuals into groups that are approximately chunksize long.
    groups = [individuals[start:(start+chunksize)] for start in range(0, len(individuals), chunksize)]
    return groups


# @jit(nopython=True, nogil=True) 
@jit(nopython=True, nogil=True) 
def phase_group(individuals, haplotype_library, set_haplotypes, n_samples):
    # Phases a group of individuals.
    for i in range(len(individuals)):
        phase(individuals[i], haplotype_library, set_haplotypes = set_haplotypes, n_samples = n_samples)


@jit(nopython=True, nogil=True) 
def phase(ind, haplotype_library, set_haplotypes, n_samples) :
    # Phases a specific individual.
    # Set_haplotypes determines whether or not to actually set the haplotypes of an individual based on the underlying samples.
    # Set_haploypes also determines whether forward_geno_probs gets calculated.

    # FLAG: Rate is hard coded.
    # FLAG: error_rate is hard coded.

    # FLAG: Do we need to check for genotype calling?

    nLoci = len(ind.genotypes)
    rate = 5/nLoci

    error_rate = 0.01

    sample_container = PhasingObjects.PhasingSampleContainer(haplotype_library, ind)
    
    calculate_forward_estimates = set_haplotypes
    track_hap_info = False

    for i in range(n_samples):
        sample_container.add_sample(rate, error_rate, calculate_forward_estimates, track_hap_info)

    pat_hap, mat_hap = sample_container.get_consensus(50)
  
    if set_haplotypes:
        add_haplotypes_to_ind(ind, pat_hap, mat_hap)

        ind.backward[:,:] = 0
        for sample in sample_container.samples:
            ind.backward += sample.forward.forward_geno_probs # We're really just averaging over particles. 

    # Always set current_haplotype after the last round of phasing.    
    ind.current_haplotypes[0][:] = pat_hap
    ind.current_haplotypes[1][:] = mat_hap
    

@jit(nopython=True, nogil=True) 
def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
    # Sets all loci to the new values (this will call missing loci as well.
    nLoci = len(pat_hap)
    ind.haplotypes[0][:] = pat_hap[:]
    ind.haplotypes[1][:] = mat_hap[:]
    ind.genotypes[:] = pat_hap[:] + mat_hap[:]
