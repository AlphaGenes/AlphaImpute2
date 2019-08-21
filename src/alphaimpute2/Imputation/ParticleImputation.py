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

from . import hap_merging

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

try:
    profile
except:
    def profile(x): 
        return x

@time_func("Runing imputation")
def impute_individuals_with_bw_library(individuals, haplotype_library):
    
    for ind in individuals:
        # We never call genotypes so can do this once.
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)

    jit_individuals = [ind.phasing_view for ind in individuals]

    loci = get_loci(jit_individuals, .5)
    print(loci)
    haplotype_library.setup_library(loci)
    for ind in jit_individuals:
        print(ind.idn)
        # impute(ind, haplotype_library.library)
        impute_hap_merge(ind, haplotype_library.library)


@jit(nopython=True, nogil=True) 
def get_loci(individuals, threshold):
    loci = None
    nInd = len(individuals)
    n_max_loci = len(individuals[0].genotypes)

    for i in range(n_max_loci):
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

@jit(nopython=True)
def impute(ind, bw_library) :
    rate = 1/bw_library.nLoci

    samples = PhasingObjects.PhasingSampleContainer(bw_library, ind)
    for i in range(100):
        samples.add_sample(rate, 0.01)

    converted_samples = [expand_sample(ind, sample, bw_library) for sample in samples.samples]

    sample_container = PhasingObjects.PhasingSampleContainer(bw_library, ind)
    sample_container.samples = converted_samples

    pat_hap, mat_hap = sample_container.get_consensus(50)
    
    # We only set a very small set of loci with the forward_geno_probs, and so need to just update our estimates of those loci and keep the rest at a neutral value.
    for index in bw_library.loci:
        ind.forward[:, index] = 0
        for sample in samples.samples:
            ind.forward[:, index] += sample.forward.forward_geno_probs[:, index] # We're really just averaging over particles. 

    add_haplotypes_to_ind(ind, pat_hap, mat_hap)

# @jit(nopython=True)
def impute_hap_merge(ind, bw_library) :
    rate = 1/bw_library.nLoci
    # print("hello world 1")
    samples = PhasingObjects.PhasingSampleContainer(bw_library, ind)
    for i in range(100):
        samples.add_sample(rate, 0.01)

    # print("hello world 2")
    sample_container = hap_merging.backwards_sampling(samples, bw_library)

    pat_hap, mat_hap = sample_container.get_consensus(50)
    
    add_haplotypes_to_ind(ind, pat_hap, mat_hap)



@jit(nopython=True, nogil=True) 
def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
    nLoci = len(pat_hap)
    for i in range(nLoci):
        # This is weird becuase it was designed to have an option to not fill in missing genotypes. 
        # It looks like filling in missing genotypes is okay though.
        ind.haplotypes[0][i] = pat_hap[i]
        ind.haplotypes[1][i] = mat_hap[i]
        ind.genotypes[i] = pat_hap[i] + mat_hap[i]



@jit(nopython=True, nogil=True)
def expand_sample(ind, sample, bw_library):

    nLoci = len(ind.genotypes)

    pat_hap = np.full(nLoci, 0, dtype = np.int8)
    mat_hap = np.full(nLoci, 0, dtype = np.int8)

    hap_info = sample.hap_info

    ranges = []
    for i in range(len(hap_info.pat_ranges)):
        range_object = hap_info.pat_ranges[i]
        global_start, global_stop = hap_info.get_global_bounds(i, 0)

        encoding_index = range_object.encoding_index

        # Select a random haplotype.
        bw_index = random.randrange(range_object.hap_range[0], range_object.hap_range[1])
        haplotype_index = bw_library.a[bw_index, encoding_index]
        pat_hap[global_start:global_stop] = bw_library.full_haps[haplotype_index, global_start:global_stop]

        ranges.append((global_start, global_stop, range_object.hap_range[0], range_object.hap_range[1]))

    for i in range(len(hap_info.mat_ranges)):
        range_object = hap_info.mat_ranges[i]
        global_start, global_stop = hap_info.get_global_bounds(i, 1)

        encoding_index = range_object.encoding_index

        # Select a random haplotype.
        bw_index = random.randrange(range_object.hap_range[0], range_object.hap_range[1])
        haplotype_index = bw_library.a[bw_index, encoding_index]
        mat_hap[global_start:global_stop] = bw_library.full_haps[haplotype_index, global_start:global_stop]

    new_sample = PhasingObjects.PhasingSample(sample.rate, sample.error_rate)
    new_sample.haplotypes = (pat_hap, mat_hap)
    new_sample.genotypes = pat_hap +  mat_hap

    new_rec = np.full(nLoci, 0, dtype = np.float32)
    for i in range(len(sample.rec)):
        true_index = bw_library.get_true_index(i)
        new_rec[true_index] = sample.rec[i]

    new_sample.rec = new_rec

    return new_sample

@jit(nopython=True, nogil=True) 
def count_differences(hap1, hap2):
    count = 0
    for i in range(len(hap1)):
        if hap1[i] != 9 and hap2[i] != 9 and hap1[i] != hap2[i]:
            count += 1
    return count
