import numba
import numpy as np
import random
import concurrent.futures

from numba import njit, jit, jitclass, prange
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

@time_func("Runing imputation")
def impute_individuals_with_bw_library(individuals, haplotype_library, n_samples):
    
    # fill in genotype probabilities.
    for ind in individuals:
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)

    jit_individuals = [ind.phasing_view for ind in individuals]

    missing_threshold = 0.5
    loci = get_non_missing_loci(jit_individuals, missing_threshold)
    print(loci)
    
    # Sets up the haplotype reference library using only the loci in loci. 
    haplotype_library.setup_library(loci)
    impute_group(jit_individuals, haplotype_library.library, n_samples)


@jit(nopython=True, nogil=True, parallel = True) 
def impute_group(individuals, library, n_samples):
    for i in range(len(individuals)):
        impute(individuals[i], library, n_samples)


@jit(nopython=True, nogil=True) 
def get_non_missing_loci(individuals, threshold):
    # Figure out which loci have a number of missing markers less than threshold.
    loci = None
    nInd = len(individuals)
    nLoci = len(individuals[0].genotypes)

    for i in range(nLoci):
        count = 0
        for j in range(nInd):
            if individuals[j].genotypes[i] != 9:
                count +=1

        if count > threshold*nInd:
            if loci is None:
                loci = [i]
            else:
                loci += [i]

    return loci

@jit(nopython=True, nogil=True) 
def impute(ind, bw_library, n_samples) :

    # FLAG: Hard coded rate. Is this even the right rate?
    rec_rate = 1/bw_library.nLoci

    calculate_forward_estimates = True
    track_hap_info = True


    sample_container = PhasingObjects.PhasingSampleContainer(bw_library, ind)
    for i in range(n_samples):
        sample_container.add_sample(rec_rate, 0.01, calculate_forward_estimates, track_hap_info)

    # This is probably more complicated than it needs to be.
    converted_samples = [expand_sample(ind, sample, bw_library) for sample in sample_container.samples]

    extended_sample_container = PhasingObjects.PhasingSampleContainer(bw_library, ind)
    extended_sample_container.samples = converted_samples

    pat_hap, mat_hap = extended_sample_container.get_consensus(50)
    
    # We only set a very small set of loci with the forward_geno_probs, and so need to just update our estimates of those loci and keep the rest at a neutral value.
    for index in bw_library.loci:
        ind.forward[:, index] = 0
        for sample in sample_container.samples:
            ind.forward[:, index] += sample.forward.forward_geno_probs[:, index] # We're really just averaging over particles. 

    add_haplotypes_to_ind(ind, pat_hap, mat_hap)


@jit(nopython=True, nogil=True) 
def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
    # DUPLICATED FROM PARTICLE PHASING

    nLoci = len(pat_hap)
    ind.haplotypes[0][:] = pat_hap[:]
    ind.haplotypes[1][:] = mat_hap[:]
    ind.genotypes[:] = pat_hap[:] + mat_hap[:]



@jit(nopython=True, nogil=True)
def expand_sample(ind, sample, bw_library):

    nLoci = len(ind.genotypes)

    pat_hap = np.full(nLoci, 9, dtype = np.int8)
    mat_hap = np.full(nLoci, 9, dtype = np.int8)

    hap_info = sample.hap_info

    # Yeah, this is all pretty ugly.
    for i in range(len(hap_info.pat_ranges)):
        range_object = hap_info.pat_ranges[i]
        global_start, global_stop = hap_info.get_global_bounds(i, 0)

        set_hap_from_range(range_object, pat_hap, global_start, global_stop, bw_library)

    for i in range(len(hap_info.mat_ranges)):
        range_object = hap_info.mat_ranges[i]
        global_start, global_stop = hap_info.get_global_bounds(i, 1)

        set_hap_from_range(range_object, mat_hap, global_start, global_stop, bw_library)

    new_sample = PhasingObjects.PhasingSample(sample.rec_rate, sample.error_rate)
    new_sample.haplotypes = (pat_hap, mat_hap)
    new_sample.genotypes = pat_hap +  mat_hap

    
    new_sample.rec = expand_rec_tracking(sample.rec, bw_library)

    return new_sample


@jit(nopython=True, nogil=True)
def expand_rec_tracking(partial_rec, bw_library):
    new_rec = np.full(bw_library.full_nLoci, 0, dtype = np.float32)
    for i in range(len(partial_rec)):
        true_index = bw_library.get_true_index(i)
        new_rec[true_index] = partial_rec[i]

    return new_rec

@jit(nopython=True, nogil=True)
def set_hap_from_range(range_object, hap, global_start, global_stop, bw_library):
    encoding_index = range_object.encoding_index

    # Select a random haplotype.
    bw_index = random.randrange(range_object.hap_range[0], range_object.hap_range[1])
    haplotype_index = bw_library.a[bw_index, encoding_index]
    hap[global_start:global_stop] = bw_library.full_haps[haplotype_index, global_start:global_stop]

