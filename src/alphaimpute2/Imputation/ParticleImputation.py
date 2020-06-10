import numba
import numpy as np
import random
import concurrent.futures

from numba.typed import List
from numba import njit, jit, jitclass, prange

from collections import OrderedDict
from itertools import repeat

from . import BurrowsWheelerLibrary
from . import PhasingObjects
from . import ParticlePhasing

from ..tinyhouse.Utils import time_func
from ..tinyhouse import InputOutput

try:
    profile
except:
    def profile(x): 
        return x


@time_func("Chip Imputation")
def impute_individuals_on_chip(ld_individuals, args, haplotype_library):

    forward_loci, reverse_loci = get_non_missing_loci(ld_individuals, 0.9)

    print("Number of individuals:", len(ld_individuals))
    print(f"Number of markers: {len(forward_loci)}")

    reverse_library = ParticlePhasing.get_reference_library(haplotype_library, reverse = True)
    reverse_library.setup_library(loci = reverse_loci, create_a = True)
    multi_threaded_apply(backward_impute_individual, [ind.reverse_individual() for ind in ld_individuals], reverse_library, args.n_imputation_particles)
    reverse_library = None
    

    forward_library = ParticlePhasing.get_reference_library(haplotype_library)
    forward_library.setup_library(loci = forward_loci, create_a = True)
    multi_threaded_apply(forward_impute_individual, ld_individuals, forward_library, args.n_imputation_particles)



def multi_threaded_apply(func, individuals, library, n_particles):

    if InputOutput.args.maxthreads <= 1:
        for ind in individuals:
            func(ind, library, n_particles)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=InputOutput.args.maxthreads) as executor:
            executor.map(func, individuals, repeat(library), 
                                            repeat(n_particles))


def backward_impute_individual(reverse_individual, library, n_samples):
    reverse_individual.setPhasingView()
    ParticlePhasing.phase(reverse_individual, library, set_haplotypes = False, imputation = True, n_samples = n_samples)

    individual = reverse_individual.reverse_view
    individual.add_backward_info()
    individual.clear_reverse_view()

def forward_impute_individual(individual, library, n_samples):
    individual.setPhasingView()
    ParticlePhasing.phase(individual, library, set_haplotypes = False, imputation = True, n_samples = n_samples)
    individual.clear_phasing_view(keep_current_haplotypes = False)


# @time_func("Imputation")
# def impute_individuals_with_bw_library(individuals, haplotype_library, n_samples):
    
#     # fill in genotype probabilities.
#     for ind in individuals:
#         ind.phasing_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0.01)

#     jit_individuals = List([ind.phasing_view for ind in individuals])

#     non_missing_threshold = 0.9
#     loci = get_non_missing_loci(jit_individuals, non_missing_threshold)
#     # print("Loci list:", loci)
    
#     # Sets up the haplotype reference library using only the loci in loci. 
#     haplotype_library.setup_library(loci)

#     if InputOutput.args.maxthreads <= 1 or len(individuals) < chunksize:
#         impute_group(jit_individuals, haplotype_library.library, n_samples)

#     else:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=InputOutput.args.maxthreads) as executor:
#             groups = split_individuals_into_groups(jit_individuals, chunksize)
#             executor.map(impute_group, groups, repeat(haplotype_library.library), repeat(n_samples))


# def split_individuals_into_groups(individuals, chunksize):
#     # Split out a list of individuals into groups that are approximately chunksize long.
#     groups = [individuals[start:(start+chunksize)] for start in range(0, len(individuals), chunksize)]
#     return groups


# @jit(nopython=True, nogil=True) 
# def impute_group(individuals, library, n_samples):
#     for i in range(len(individuals)):
#         ParticlePhasing.phase(individuals[i], library, set_haplotypes = False, imputation = True, n_samples = n_samples)

def get_non_missing_loci(individuals, threshold):
    nLoci = len(individuals[0].genotypes)
    scores = np.full(nLoci, 0, dtype = np.float32)
    for ind in individuals:
        scores += (ind.genotypes != 9)
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


# @jit(nopython=True, nogil=True) 
# def get_non_missing_loci(individuals, threshold):
#     # Figure out which loci have a number of missing markers less than threshold.
#     loci = None
#     nInd = len(individuals)
#     nLoci = len(individuals[0].genotypes)

#     for i in range(nLoci):
#         count = 0
#         for j in range(nInd):
#             if individuals[j].genotypes[i] != 9:
#                 count +=1
#         if count > threshold*nInd:
#             if loci is None:
#                 loci = [i]
#             else:
#                 loci += [i]

#     return loci

# @jit(nopython=True, nogil=True) 
# def impute(ind, bw_library, n_samples) :

#     # FLAG: Hard coded rate. Is this even the right rate?
#     rec_rate = 1/bw_library.nLoci

#     calculate_forward_estimates = True
#     track_hap_info = True


#     sample_container = PhasingObjects.PhasingSampleContainer(bw_library, ind)
#     for i in range(n_samples):
#         sample_container.add_sample(rec_rate, 0.01, calculate_forward_estimates, track_hap_info)

#     # This is probably more complicated than it needs to be.
#     converted_samples = [expand_sample(ind, sample, bw_library) for sample in sample_container.samples]

#     extended_sample_container = PhasingObjects.PhasingSampleContainer(bw_library, ind)
#     extended_sample_container.samples = converted_samples

#     pat_hap, mat_hap = extended_sample_container.get_consensus(50)
    
#     # We only set a very small set of loci with the forward_geno_probs, and so need to just update our estimates of those loci and keep the rest at a neutral value.
    
#     backward = ind.backward # Not sure why we need to do this, but it turns out we do.
#     for index in bw_library.loci:
#         for j in range(4):
#             backward[j, index] = 0.0

#         for sample in sample_container.samples:
#             for j in range(4):
#                 backward[j, index] += sample.forward.forward_geno_probs[j, index] # We're really just averaging over particles. 

#     add_haplotypes_to_ind(ind, pat_hap, mat_hap)


# @jit(nopython=True, nogil=True) 
# def add_haplotypes_to_ind(ind, pat_hap, mat_hap):
#     # DUPLICATED FROM PARTICLE PHASING

#     nLoci = len(pat_hap)
#     ind.haplotypes[0][:] = pat_hap[:]
#     ind.haplotypes[1][:] = mat_hap[:]
#     ind.genotypes[:] = pat_hap[:] + mat_hap[:]



# @jit(nopython=True, nogil=True)
# def expand_sample(ind, sample, bw_library):

#     nLoci = len(ind.genotypes)

#     pat_hap = np.full(nLoci, 9, dtype = np.int8)
#     mat_hap = np.full(nLoci, 9, dtype = np.int8)

#     hap_info = sample.hap_info

#     # Yeah, this is all pretty ugly.
#     for i in range(len(hap_info.pat_ranges)):
#         range_object = hap_info.pat_ranges[i]
#         global_start, global_stop = hap_info.get_global_bounds(i, 0)

#         set_hap_from_range(range_object, pat_hap, global_start, global_stop, bw_library)

#     for i in range(len(hap_info.mat_ranges)):
#         range_object = hap_info.mat_ranges[i]
#         global_start, global_stop = hap_info.get_global_bounds(i, 1)

#         set_hap_from_range(range_object, mat_hap, global_start, global_stop, bw_library)

#     new_sample = PhasingObjects.PhasingSample(sample.rec_rate, sample.error_rate)
#     new_sample.haplotypes = (pat_hap, mat_hap)
#     new_sample.genotypes = pat_hap +  mat_hap

    
#     new_sample.rec = expand_rec_tracking(sample.rec, bw_library)

#     return new_sample


# @jit(nopython=True, nogil=True)
# def expand_rec_tracking(partial_rec, bw_library):
#     new_rec = np.full(bw_library.full_nLoci, 0, dtype = np.float32)
#     for i in range(len(partial_rec)):
#         true_index = bw_library.get_true_index(i)
#         new_rec[true_index] = partial_rec[i]

#     return new_rec

# @jit(nopython=True, nogil=True)
# def set_hap_from_range(range_object, hap, global_start, global_stop, bw_library):
#     encoding_index = range_object.encoding_index

#     # Select a random haplotype.
    
#     # if range_object.hap_range[0] == range_object.hap_range[1]:
#     # print(range_object.hap_range, range_object.start, range_object.stop, range_object.encoding_index)
#     bw_index = random.randrange(range_object.hap_range[0], range_object.hap_range[1])
#     haplotype_index = bw_library.a[bw_index, encoding_index]
#     hap[global_start:global_stop] = bw_library.full_haps[haplotype_index, global_start:global_stop]

