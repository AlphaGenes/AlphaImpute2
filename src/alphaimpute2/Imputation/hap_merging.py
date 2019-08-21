from .PhasingObjects import PhasingSampleContainer, PhasingSample
import numpy as np
import numba
from numba import jit, njit, jitclass
from collections import OrderedDict

try:
    profile
except:
    def profile(x): 
        return x

@profile
def backwards_sampling(sample_container, bw_library):
    # print("hello world 3")

    nLoci = bw_library.nLoci
    nSamples = len(sample_container.samples)
    pat_lower_bounds = np.full((nSamples, nLoci), 0, dtype = np.int64)
    pat_upper_bounds = np.full((nSamples, nLoci), 0, dtype = np.int64)

    mat_lower_bounds = np.full((nSamples, nLoci), 0, dtype = np.int64)
    mat_upper_bounds = np.full((nSamples, nLoci), 0, dtype = np.int64)
    # print("hello world 4")

    for sample_index, sample in enumerate(sample_container.samples):
        pat_lower_bounds[sample_index, :] = sample.forward.pat_ranges[0,:]
        pat_upper_bounds[sample_index, :] = sample.forward.pat_ranges[1,:]
        
        mat_lower_bounds[sample_index, :] = sample.forward.mat_ranges[0,:]
        mat_upper_bounds[sample_index, :] = sample.forward.mat_ranges[1,:]

    particle_info = ParticleInformation(pat_lower_bounds, pat_upper_bounds, mat_lower_bounds, mat_upper_bounds)
    
    # print("hello world 5")
    backwards_samples = [stochastic_search_long(particle_info, bw_library) for i in range(100)]

    new_sample_container = PhasingSampleContainer(sample_container.bw_library, sample_container.ind)
    new_sample_container.samples = backwards_samples
    return new_sample_container

@jit(nopython=True, nogil=True) 
def stochastic_search_long(particle_info, bw_library):
    # I'm doing this badly.
    nLoci = bw_library.nLoci
    rec_rate = 5/nLoci

    index = nLoci -1
    pat_hap, mat_hap = sample_joint_hap(index, particle_info, bw_library)

    pat_states = np.full(nLoci, 0, dtype = np.int64)
    mat_states = np.full(nLoci, 0, dtype = np.int64)

    values = np.full(4, 0, dtype = np.float32)

    # print("hello world 6")

    pat_states[nLoci-1] = pat_hap
    mat_states[nLoci-1] = mat_hap

    for index in range(nLoci -2, -1, -1): # Iterate backwards
        # Check for recombination.
        # print("hello world 7")
    
        p_both = (1-rec_rate)**2 * get_joint_prob(pat_hap, mat_hap, index, particle_info, bw_library)
        # print("hello world 8")

        p_pat = (1-rec_rate)*(rec_rate) * get_single_prob(pat_hap, index, 0, particle_info, bw_library)

        # print("hello world 9")
        p_mat = (1-rec_rate)*(rec_rate) * get_single_prob(mat_hap, index, 1, particle_info, bw_library)
        p_none = (rec_rate)**2

        # print("hello world 10")

        # Choose one of these options. 

        values[0] = p_both
        values[1] = p_pat
        values[2] = p_mat
        values[3] = p_none
        
        # print(index, pat_hap, mat_hap, values)

        option = sample_1D(values)
        
        # print(index, pat_hap, mat_hap, values, option)


        if option == 0 : # No recombination.
            pat_hap = pat_hap
            mat_hap = mat_hap

        if option == 1 : # no paternal recombination; maternal recombination.
            pat_hap = pat_hap
            mat_hap = sample_single_hap(index, pat_hap, 0, particle_info, bw_library)
        
        if option == 2 : # no paternal recombination; maternal recombination.
            pat_hap = sample_single_hap(index, mat_hap, 1, particle_info, bw_library)
            mat_hap = mat_hap

        if option == 3 : # no paternal recombination; maternal recombination.
            pat_hap, mat_hap = sample_joint_hap(index, particle_info, bw_library)

        pat_states[index] = pat_hap
        mat_states[index] = mat_hap

    output_pat_hap = extract_full_haplotype(pat_states, bw_library)
    output_mat_hap = extract_full_haplotype(mat_states, bw_library)

    new_sample = PhasingSample(rec_rate, 0.01)

    new_sample.haplotypes = (output_pat_hap, output_mat_hap)
    new_sample.genotypes = output_pat_hap + output_mat_hap
    new_sample.rec = np.full(bw_library.full_nLoci, 0, dtype = np.float32)

    return new_sample

@jit(nopython=True, nogil=True) 
def extract_full_haplotype(haplotype_list, bw_library):
    nLoci = bw_library.nLoci
    full_nLoci = bw_library.full_nLoci

    start = 0
    current_hap = haplotype_list[0]

    output_haplotype = np.full(full_nLoci, 9, dtype = np.int8)
    for i in range(nLoci):
        if haplotype_list[i] != current_hap:
            # Recombination here. Fill in haplotypes.
            stop = int(np.ceil((bw_library.get_true_index(i-1) + bw_library.get_true_index(i))/2))

            # print(start, stop, current_hap)
            output_haplotype[start:stop] = bw_library.full_haps[current_hap, start:stop]

            start = stop
            current_hap = haplotype_list[i]
    # And handle the end case.
    stop = full_nLoci
    output_haplotype[start:stop] = bw_library.full_haps[current_hap, start:stop]

    return output_haplotype

@jit(nopython=True, nogil=True) 
def get_joint_prob(pat_hap, mat_hap, index, particle_info, bw_library):
    bw_pat = normal_to_bw(bw_library, pat_hap, index)
    bw_mat = normal_to_bw(bw_library, mat_hap, index)

    weight = 0
    for particle_index in range(particle_info.nParticles):

        if particle_info.contains(particle_index, index, bw_pat, 0) and particle_info.contains(particle_index, index, bw_mat, 1):
            value =  1.0/(particle_info.size(particle_index, index, 0) * particle_info.size(particle_index, index, 1))
            weight += 1
    return weight

@jit(nopython=True, nogil=True) 
def get_single_prob(hap, index, hap_is_mat, particle_info, bw_library):
    # print(hap, index, bw_library.reverse_library.shape, particle_info.nParticles)
    bw_hap = normal_to_bw(bw_library, hap, index)
    weight = 0
    # print(bw_hap)
    for particle_index in range(particle_info.nParticles):
        # print(particle_index)
        if particle_info.contains(particle_index, index, bw_hap, hap_is_mat) :
            value = 1.0/(particle_info.size(particle_index, index, hap_is_mat))
            weight += 1.0
    return weight


@jit(nopython=True, nogil=True) 
def sample_joint_hap(index, particle_info, bw_library):
    particle_index = np.random.randint(particle_info.nParticles)
    bw_pat, bw_mat = particle_info.sample_joint_hap(particle_index, index)
    pat_hap = bw_to_normal(bw_library, bw_pat, index)
    mat_hap = bw_to_normal(bw_library, bw_mat, index)

    return pat_hap, mat_hap


@jit(nopython=True, nogil=True) 
def sample_single_hap(index, hap, hap_is_mat, particle_info, bw_library):
    nParticles = particle_info.nParticles
    weights = np.full(nParticles, 0, dtype = np.float32)

    bw_hap = normal_to_bw(bw_library, hap, index)
    for particle_index in range(nParticles):
        if particle_info.contains(particle_index, index, bw_hap, hap_is_mat):
            weights[particle_index] = particle_info.size(particle_index, index, hap_is_mat)

    particle_index = sample_1D(weights)

    sampled_bw_hap = particle_info.sample_single_hap(particle_index, index, 1-hap_is_mat) # Sample something from the other parent.

    return bw_to_normal(bw_library, sampled_bw_hap, index)

@jit(nopython=True, nogil=True) 
def normal_to_bw(bw_library, hap, index):
    return bw_library.reverse_library[hap, index]

@jit(nopython=True, nogil=True) 
def bw_to_normal(bw_library, hap, index):
    return bw_library.a[hap, index]




######################
#
#      CLASS         #
#
######################

spec = OrderedDict()

spec['pat_lower_bounds'] = numba.int64[:,:]
spec['pat_upper_bounds'] = numba.int64[:,:]
spec['mat_lower_bounds'] = numba.int64[:,:]
spec['mat_upper_bounds'] = numba.int64[:,:]
spec['nParticles'] = numba.int64

@jitclass(spec)
class ParticleInformation(object):
    # All of this is in the context of a specific library.
    def __init__(self, pat_lower_bounds, pat_upper_bounds, mat_lower_bounds, mat_upper_bounds):
        self.nParticles = pat_lower_bounds.shape[0]
        self.pat_lower_bounds = pat_lower_bounds        
        self.pat_upper_bounds = pat_upper_bounds        
        self.mat_lower_bounds = mat_lower_bounds
        self.mat_upper_bounds = mat_upper_bounds

    def get_bounds(self, is_mat):
        if is_mat:
            return self.mat_lower_bounds, self.mat_upper_bounds
        else:
            return self.pat_lower_bounds, self.pat_upper_bounds

    def contains(self, particle_index, index, hap, is_mat):
        lower, upper = self.get_bounds(is_mat)
        # print(particle_index, index, lower.shape, upper.shape)
        if lower[particle_index, index] <= hap and hap < upper[particle_index, index] :
            return True
        else:
            return False

    
    def sample_joint_hap(self, particle_index, index):
        pat_hap = self.sample_single_hap(particle_index, index, 0)
        mat_hap = self.sample_single_hap(particle_index, index, 1)

        return pat_hap, mat_hap


    def sample_single_hap(self, particle_index, index, is_mat):
        lower, upper = self.get_bounds(is_mat)
        lower_i = lower[particle_index, index] 
        upper_i = upper[particle_index, index] 
        return np.random.randint(lower_i, upper_i)


    def size(self, particle_index, index, is_mat):
        lower, upper = self.get_bounds(is_mat)
        return upper[particle_index, index] - lower[particle_index, index] 


######################
#
#      UTILITIES     #
#
######################


# @jit(nopython=True, nogil=True) 
# def sample_index_greater_than_zero(vect):
#     nParticles = len(vect)
#     nActive = 0
#     for j in range(nParticles):
#         if vect[j] > 0:
#             nActive += 1

#     count = np.random.random()*nActive
#     index = -1
#     while count > 0:
#         index += 1
#         if vect[index] > 0:
#             count -= 1

#     return index

@jit(nopython=True, nogil=True) 
def sample_1D(vect):
    nVect = len(vect)
    total = 0.0
    for j in range(nVect):
        total += vect[j]

    value = np.random.random()*total
    index = -1
    while value > 0:
        index += 1
        value -= vect[index]
    return index
