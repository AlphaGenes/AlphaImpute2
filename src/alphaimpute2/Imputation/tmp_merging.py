
# Alternative way which exploits long periods of the same support from individual particles.
# Could we run an HMM in particle space?
# Yeaaaaah... it would be... quadratic in terms of number of particles?
# The coding of groups at each loci is a little bit... wonky.
# Individual groups _do_not_ get pushed back nicely in terms of the library.
    # Intuition on this: When building the library, all of the (0, 1), (0, 0) states get bundled together at locus 0.
    # When we push those states forward, the (0,1) states and the (0,0) states seperate, but without a clean breakpoint.
# I think this should be done on individual haplotype levels.

def stochastic_search_long(particle_library, bw_library):
    rate = 1/nLoci

    current_index = nLoci -1
    # Get initial set of haplotypes.
    pat_hap, mat_hap, pair_length = update(-1, -1, True, True, current_index)

    pat_hap = np.full(nLoci, 0, dtype = np.int8)
    mat_hap = np.full(nLoci, 0, dtype = np.int8)

    while current_index > 0:
        pat_hap, mat_hap, pair_length = update(-1, -1, True, True, current_index)

        rec_pat = np.random.poisson(rate)+1
        rec_mat = np.random.poisson(rate)+1

        if rec_pat > pair_length and rec_mat > pair_length:
            length = pair_length
            new_pat = False
            new_mat = False

        elif rec_pat < rec_mat:
            length = rec_pat
            new_pat = True

        elif rec_mat < rec_pat:
            length = rec_pat
            new_pat = True

        elif rec_mat == rec_pat:
            length = rec_pat
            new_pat = True
            new_mat = True 

        # Do we need to catch something where start + stop = 0?
        start = min(current_index + 1, nLoci)
        stop = max(current_index - length + 1, 0)

        # Fill in the paternal and the maternal haplotypes.
        pat_hap[start:stop] = bw_library.haps[haplotype_pair[0], start:stop]
        mat_hap[start:stop] = bw_library.haps[haplotype_pair[1], start:stop]

        current_index -= length


def update(pat_hap, mat_hap, new_pat, new_mat, current_index):

    # Get haplotypes that have support for the current particle

    paternal_support = np.full(nParticles, 0, dtype = np.int8)
    maternal_support = np.full(nParticles, 0, dtype = np.int8)
    joint_support = np.full(nParticles, 0, dtype = np.int8)

    # Get initial support to determine which particles can hold which haplotypes.
    for particle in particle_library.particles:
        paternal_support[i], maternal_support[i], joint_support[i] = get_support(particle, pat_hap, mat_hap, current_index)

    # Update haplotypes
    if new_pat and new_mat:
        index = np.random.randrange(len(particle_library.particles))
        particle = particle_library.particles[index]
        pat_hap, mat_hap = particle.sample_haplotype_pair()

    elif new_pat:
        # Looking for a new paternal haplotype. Sample haplotypes from particles with _maternal_ support 
        index = sample_non_zero(maternal_support)
        particle = particle_library.particles[index]
        pat_hap = particle.sample_pat_hap()

    elif new_mat:
        # Looking for a new maternal haplotype. Sample haplotypes from particles with _paternal_ support 
        index = sample_non_zero(maternal_support)
        particle = particle_library.particles[index]
        mat_hap = particle.sample_mat_hap()

    for particle in particle_library.particles:
        paternal_support[i], maternal_support[i], joint_support[i] = get_support(particle, pat_hap, mat_hap, current_index)
        # Get haplotypes with longest joint_support

    length = np.max(joint_support)

    return pat_hap, mat_hap, length


def get_support(particle, pat_hap, mat_hap, index):

    encoding_index = particle.encoding[index]

    encoded_pat_hap = bw_library.reverse_library[pat_hap, encoding_index]
    encoded_mat_hap = bw_library.reverse_library[mat_hap, encoding_index]

    # Need to figure out how I want to make indexes work.

    if encoded_pat_hap >= particle.lower[index] and encoded_mat_hap < particle.upper[index]:
        pat_support = particle.haplotype_length[index]

    if encoded_mat_hap >= particle.lower[index] and encoded_mat_hap < particle.upper[index]:
        mat_support = particle.haplotype_length[index]

    joint_support = min(pat_support, mat_support)
    return pat_support, mat_support, joint_support



class Particle(object):

    def __init__(self, hap_info, nLoci):

        self.nLoci = nLoci
        self.lower = np.full(nLoci, -1, dtype = np.int64) # Lower bound of the range
        self.upper = np.full(nLoci, -1, dtype = np.int64) # Upper bound of the range

        self.haplotype_length = np.full(nLoci, -1, dtype = np.int64)
        self.encoding = np.full(nLoci, -1, dtype = np.int64) # Which index is the range encoded in?



@jit(nopython=True, nogil=True) 
def weighted_sample_1D(mat):
    # total = np.sum(mat)
    
    total = 0
    for i in range(mat.shape[0]):
        total += mat[i]
    value = random.random()*total

    for i in range(mat.shape[0]):
        value -= mat[i]
        if value < 0:
            return i


    return 0

