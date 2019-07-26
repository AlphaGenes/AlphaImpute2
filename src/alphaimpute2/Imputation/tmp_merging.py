
# Alternative way which exploits long periods of the same support from individual particles.
# Could we run an HMM in particle space?
# Yeaaaaah... it would be... quadratic in terms of number of particles?
# The coding of groups at each loci is a little bit... wonky.
# Individual groups _do_not_ get pushed back nicely in terms of the library.
    # Intuition on this: When building the library, all of the (0, 1), (0, 0) states get bundled together at locus 0.
    # When we push those states forward, the (0,1) states and the (0,0) states seperate, but without a clean breakpoint.
# I think this should be done on individual haplotype levels.

from .PhasingObjects import PhasingSampleContainer, PhasingSample
import numpy as np
from numba import jit

try:
    profile
except:
    def profile(x): 
        return x

@profile
def backwards_sampling(sample_container):
    print(sample_container.ind.idn)
    for sample in sample_container.samples:
        sample.hap_info.setup_global_matrix()

    vals = [sample.hap_info.get_mean_haplotype_length() for sample in sample_container.samples]
    print(vals)

    particle_library = [sample.hap_info for sample in sample_container.samples]

    new_sample_container = PhasingSampleContainer(sample_container.bw_library, sample_container.ind)
    
    backwards_samples = [stochastic_search_long(particle_library, sample_container.bw_library) for i in range(40)]

    new_sample_container.samples = backwards_samples
    return new_sample_container

@jit(nopython=True, nogil=True) 
def stochastic_search_long(particle_library, bw_library):
    nLoci = bw_library.full_nLoci
    rate = 5/nLoci

    index = nLoci -1
    # Get initial set of haplotypes.
    pat_hap, mat_hap = sample_random_hap(particle_library, index)

    output_pat_hap = np.full(nLoci, 0, dtype = np.int8)
    output_mat_hap = np.full(nLoci, 0, dtype = np.int8)


    hap_end_count = 0
    while index >= 0:
        pat_support, mat_support, joint_support = check_support(pat_hap, mat_hap, index, particle_library)
        joint_length = np.max(joint_support)

        rec_pat = np.random.geometric(rate)
        rec_mat = np.random.geometric(rate)

        print("particle_info", index, rec_pat, rec_mat, joint_length)

        # Recombinations lie after the point where these haplotypes are valid.
        new_pat = False
        new_mat = False

        if rec_pat > joint_length and rec_mat > joint_length:
            length = joint_length

            hap_end_count +=1 


        elif rec_pat < rec_mat:
            length = rec_pat
            new_pat = True

        elif rec_mat < rec_pat:
            length = rec_mat
            new_mat = True

        elif rec_mat == rec_pat:
            length = rec_pat
            new_pat = True
            new_mat = True 

        # Do we need to catch something where start + stop = 0?
        start = max(index - length + 1, 0)
        stop = min(index + 1, nLoci)
        
        # Fill in the paternal and the maternal haplotypes.
        output_pat_hap[start:stop] = bw_library.full_haps[pat_hap, start:stop]
        output_mat_hap[start:stop] = bw_library.full_haps[mat_hap, start:stop]



        index -= length
        pat_hap, mat_hap = get_new_haps(pat_hap, mat_hap, new_mat, new_pat, index, particle_library)

    new_sample = PhasingSample(rate)

    new_sample.haplotypes = (output_pat_hap, output_mat_hap)
    new_sample.genotypes = output_pat_hap + output_mat_hap
    new_sample.rec = np.full(nLoci, 0, dtype = np.int64)

    print(hap_end_count)
    return new_sample

@jit(nopython=True, nogil=True) 
def get_new_haps(pat_hap, mat_hap, new_mat, new_pat, index, particle_library):

    paternal_support, maternal_support, joint_support = check_support(pat_hap, mat_hap, index, particle_library)

    pat_length = np.max(paternal_support)
    mat_length = np.max(maternal_support)

    if not new_mat and not new_pat:
        # This case happens when the haplotype ran out.
        # Check if each of the haplotypes are valid on their own.
        # if both are, pick one randomly.
        if pat_length == 0 and mat_length == 0:
            # Neither haplotype are valid -- re-choose both.
            new_mat = True
            new_pat = True

        if pat_length != 0 and mat_length == 0:
            # Paternal is valid. Pick a new maternal.
            new_mat = True

        if pat_length == 0 and mat_length != 0:
            new_pat = True

        if pat_length != 0 and mat_length !=0:
            # Both are valid, pick one randomly.
            if np.random.random() < .5:
                new_pat = True
            else:
                new_mat = True


    if new_mat and new_pat:
        pat_hap, mat_hap = sample_random_hap(particle_library, index)
    elif new_mat:
        pat_hap, mat_hap = sample_random_hap_with_support(pat_hap, paternal_support, particle_library, index, 0)
    elif new_pat:
        pat_hap, mat_hap = sample_random_hap_with_support(mat_hap, maternal_support, particle_library, index, 1)

    return pat_hap, mat_hap

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
