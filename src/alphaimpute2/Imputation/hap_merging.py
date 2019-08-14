from .PhasingObjects import PhasingSampleContainer, PhasingSample
import numpy as np
from numba import jit

try:
    profile
except:
    def profile(x): 
        return x

@profile
def backwards_sampling(sample_container, bw_library):

    nLoci = bw_library.nLoci

    pat_lower_bounds = np.full((nLoci, nSamples), 0, dtype = np.float32)
    pat_upper_bounds = np.full((nLoci, nSamples), 0, dtype = np.float32)

    mat_lower_bounds = np.full((nLoci, nSamples), 0, dtype = np.float32)
    mat_upper_bounds = np.full((nLoci, nSamples), 0, dtype = np.float32)

    for sample in sample_container.samples:
        pat_lower_bounds = sample.forward.pat_ranges[0,:]
        pat_upper_bounds = sample.forward.pat_ranges[1,:]
        
        mat_lower_bounds = sample.forward.mat_ranges[0,:]
        mat_upper_bounds = sample.forward.mat_ranges[1,:]

    backwards_samples = [stochastic_search_long(pat_lower_bounds, pat_upper_bounds, mat_lower_bounds, mat_upper_bounds, bw_library) for i in range(40)]

    new_sample_container.samples = backwards_samples
    return new_sample_container

@jit(nopython=True, nogil=True) 
def stochastic_search_long(particle_info, bw_library):
    # I'm doing this badly.
    nLoci = bw_library.nLoci
    rate = 5/nLoci

    index = nLoci -1
    pat_hap, mat_hap = sample_random_hap(particle_info, bw_library, index)

    output_pat_hap = np.full(nLoci, 0, dtype = np.int8)
    output_mat_hap = np.full(nLoci, 0, dtype = np.int8)

    values = np.full(4, 0, dtype = np.float32)

    for i in range(index -2, -1, -1): # Iterate backwards

        # Check for recombination.
        p_both = (1-rec)**2 * get_joint_prob(pat_hap, mat_hap, index)
        p_pat = (1-rec)*(rec) * get_single_prob(pat_hap, index, 0)
        p_mat = (1-rec)*(rec) * get_single_prob(mat_hap, index, 1)
        p_none = (rec)**2


        # Choose one of these options. 

        values[0] = p_both
        values[1] = p_pat
        values[2] = p_mat
        values[3] = p_none

        option = sample_1D(values)

        if option == 0 : # No recombination.
            pat_hap = pat_hap
            mat_hap = mat_hap

        if option == 1 : # no paternal recombination; maternal recombination.
            pat_hap = pat_hap
            mat_hap = sample_single_hap(index, pat_hap, 1)
        
        if option == 3 : # no paternal recombination; maternal recombination.
            pat_hap = sample_single_hap(index, mat_hap, 0)
            mat_hap = mat_hap

        if option == 4 : # no paternal recombination; maternal recombination.
            pat_hap, mat_hap = sample_joint_hap(particle_info, bw_library, index)

    return new_sample

@jit(nopython=True, nogil=True) 
def get_joint_prob(pat_hap, mat_hap, index, particle_info, bw_library):

    bw_pat = project(bw_library, pat_hap)
    bw_mat = project(bw_library, mat_hap)

    weight = 0
    for i in range(nSamples):
        if bw_pat >= pat_lower_bounds and bw_pat < pat_upper_bounds:
            if bw_mat >= mat_lower_bounds and 


def project(bw_library, hap):
    return bw_library.a[hap]

@jit(nopython=True, nogil=True) 
def sample_random_hap(particle_library, index):

    particle_index = np.random.randint(len(particle_library))

    pat_hap = particle_library[particle_index].get_random_haplotype(index, 0)
    mat_hap = particle_library[particle_index].get_random_haplotype(index, 1)

    return pat_hap, mat_hap

@jit(nopython=True, nogil=True) 
def sample_random_hap_with_support(ref_hap, support, particle_library, index, hap):

    particle_index = sample_index_greater_than_zero(support)
    particle = particle_library[particle_index]

    if hap == 0:
        mat_hap = particle.get_random_haplotype(index, 1)
        return ref_hap, mat_hap
    else:
        pat_hap = particle.get_random_haplotype(index, 0)
        return pat_hap, ref_hap

@jit(nopython=True, nogil=True) 
def check_support(pat_hap, mat_hap, index, particle_library):
    nParticles = len(particle_library)
    # Get haplotypes that have support for the current particle

    paternal_support = np.full(nParticles, 0, dtype = np.int64)
    maternal_support = np.full(nParticles, 0, dtype = np.int64)
    joint_support = np.full(nParticles, 0, dtype = np.int64)

    # Get initial support to determine which particles can hold which haplotypes.
    for i, particle in enumerate(particle_library):
        paternal_support[i], maternal_support[i], joint_support[i] = get_support(particle, pat_hap, mat_hap, index)

    return paternal_support, maternal_support, joint_support

@jit(nopython=True, nogil=True) 
def get_support(hap_info, pat_hap, mat_hap, index):
    pat_support = hap_info.check_inclusion(pat_hap, index, 0)
    mat_support = hap_info.check_inclusion(mat_hap, index, 1)
    return pat_support, mat_support, min(pat_support, mat_support)



@jit(nopython=True, nogil=True) 
def sample_index_greater_than_zero(vect):
    nParticles = len(vect)
    nActive = 0
    for j in range(nParticles):
        if vect[j] > 0:
            nActive += 1

    count = np.random.random()*nActive
    index = -1
    while count > 0:
        index += 1
        if vect[index] > 0:
            count -= 1

    return index

@jit(nopython=True, nogil=True) 
def sample_1D(vect):
    nVect = len(vect)
    total = 0
    for j in range(nVect):
        total += vect[j]

    value = np.random.random()*nActive
    index = -1
    while value > 0:
        index += 1
        value -= vect[index]
    return index
