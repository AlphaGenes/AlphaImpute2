import numba
import numpy as np
import random

from numba import njit, jit, jitclass
from collections import OrderedDict

from . import BurrowsWheelerLibrary
from . import Imputation
from . import ImputationIndividual

try:
    profile
except:
    def profile(x): 
        return x

spec = OrderedDict()

spec['start'] = numba.int64
spec['stop'] = numba.int64
spec['encoding_index'] = numba.int64
spec['hap_range'] = numba.typeof((0,1))

@jitclass(spec)
class HaplotypeRange(object):
    def __init__(self, start, stop, hap_range, encoding_index):
        self.start = start
        self.stop = stop

        self.encoding_index = encoding_index
        self.hap_range = hap_range

example_haplotype_range = HaplotypeRange(0, 0, (0, 0), 0)

spec = OrderedDict()

spec['pat_ranges'] = numba.optional(numba.typeof([example_haplotype_range, example_haplotype_range])) # i.e. array of HaplotypeRanges
spec['mat_ranges'] = numba.optional(numba.typeof([example_haplotype_range, example_haplotype_range])) # i.e. array of HaplotypeRanges
spec['bw_library'] = numba.typeof(BurrowsWheelerLibrary.get_example_library().library)

spec['range_index'] = numba.int64[:,:]
spec['residual_length'] = numba.int64[:,:]

@jitclass(spec)
class HaplotypeInformation(object):
    # All of this is in the context of a specific library.
    def __init__(self, bw_library):
        self.bw_library = bw_library
        
        self.pat_ranges = None    
        self.mat_ranges = None    

    def add_mat_sample(self, index, hap_range):

        if self.mat_ranges is None:
            start = 0
        else:
            start = self.mat_ranges[-1].stop # Start value is the previous stop value.

        stop = index # The haplotype goes all the way up until the next index though.
        encoding_index = index # Encoded at the index.
        new_range = HaplotypeRange(start, stop, hap_range, encoding_index)  

        if self.mat_ranges is None:
            self.mat_ranges = [new_range]
        else:
            self.mat_ranges += [new_range]

        
    def add_pat_sample(self, index, hap_range):

        if self.pat_ranges is None:
            start = 0
        else:
            start = self.pat_ranges[-1].stop # Start value is the previous stop value.

        stop = index # The haplotype goes all the way up until the next index though.
        encoding_index = index # Encoded at the index.
        new_range = HaplotypeRange(start, stop, hap_range, encoding_index)  

        if self.pat_ranges is None:
            self.pat_ranges = [new_range]
        else:
            self.pat_ranges += [new_range]

    def get_global_bounds(self, index, hap):
        if hap == 0:
            ranges = self.pat_ranges
        if hap == 1:
            ranges = self.mat_ranges

        # The 1 offset is because sub_start should be included in the previous range (and not the current range), and then we include sub_end in our current range.
        global_start = self.bw_library.get_true_index(ranges[index].start)+1

        # The 1 offset is because true_end should be included in the current range, and this is a python range so we need to go 1 further.
        global_end = self.bw_library.get_true_index(ranges[index].stop)+1

        if index == 0:
            global_start = 0
        if index == len(ranges) - 1:
            global_end = self.bw_library.full_nLoci

        return global_start, global_end


    def get_mean_haplotype_length(self):
        val = 0
        for i in range(len(self.pat_ranges)):
            global_start, global_end = self.get_global_bounds(i, 0)
            val += global_end - global_start

        return val/len(self.pat_ranges)

    def setup_global_matrix(self):
        self.range_index = np.full((2, self.bw_library.full_nLoci), -1, dtype = np.int64)
        self.residual_length = np.full((2, self.bw_library.full_nLoci), -1, dtype = np.int64)
        
        for hap in range(2):
            if hap == 0:
                ranges = self.pat_ranges
            if hap == 1:
                ranges = self.mat_ranges

            for i in range(len(ranges)):
                global_start, global_end = self.get_global_bounds(i, hap)
                self.range_index[hap, global_start:global_end] = i
                self.residual_length[hap, global_start:global_end] = np.arange(global_end - global_start) + 1


    def check_inclusion(self, ref_hap, index, hap):
        if ref_hap < 0: 
            return 0
        
        if hap == 0:
            ranges = self.pat_ranges
        if hap == 1:
            ranges = self.mat_ranges

        range_object = ranges[self.range_index[hap, index]]
        encoded_hap = self.bw_library.reverse_library[ref_hap, range_object.encoding_index]

        contains_hap = (range_object.hap_range[0] <= encoded_hap) and (encoded_hap < range_object.hap_range[1])

        if not contains_hap: 
            return 0
        else:
            return self.residual_length[hap, index]

    def get_random_haplotype(self, index, hap):

        if hap == 0:
            ranges = self.pat_ranges
        if hap == 1:
            ranges = self.mat_ranges

        range_object = ranges[self.range_index[hap, index]]
        # print(range_object.hap_range)
        encoded_hap = np.random.randint(range_object.hap_range[0], range_object.hap_range[1])
        output_hap = self.bw_library.a[encoded_hap, range_object.encoding_index]
        return output_hap


spec = OrderedDict()

# These hold the paternal and maternal ranges of a haplotype at a specific loci.
spec['pat_ranges'] = numba.int64[:,:] # 2 x n
spec['mat_ranges'] = numba.int64[:,:]

# These hold the forward genotype probabilities at a specific loci.
spec['forward_geno_probs'] = numba.float32[:,:]

@jitclass(spec)
class ForwardHaplotype(object):
    # All of this is in the context of a specific library.
    def __init__(self, nLoci, full_nLoci):
        self.pat_ranges = np.full((2, nLoci), 0, dtype = np.int64)    
        self.mat_ranges = np.full((2, nLoci), 0, dtype = np.int64)    

        self.forward_geno_probs = np.full((4, full_nLoci), 1, dtype = np.float32)

spec = OrderedDict()
spec['genotypes'] = numba.int8[:]
spec['rec'] = numba.float32[:]

spec['rate'] = numba.float32
spec['error_rate'] = numba.float32
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

tmp_info = HaplotypeInformation(BurrowsWheelerLibrary.get_example_library().library)
spec['hap_info'] = numba.typeof(tmp_info)

tmp_forward = ForwardHaplotype(10, 100)
spec['forward'] = numba.typeof(tmp_forward)

@jitclass(spec)
class PhasingSample(object):

    def __init__(self, rate, error_rate):
        self.rate = rate
        self.error_rate = error_rate

    def sample(self, bw_library, ind):
        raw_genotypes, rec, self.hap_info= self.haplib_sample(bw_library, ind, self.rate, self.error_rate)
        self.haplotypes = self.get_haplotypes(raw_genotypes)
        self.genotypes = self.haplotypes[0] + self.haplotypes[1]
        self.rec = rec

    def get_haplotypes(self, raw_genotypes):
        nLoci = len(raw_genotypes)
        pat_hap = np.full(nLoci, 9, dtype = np.int8)
        mat_hap = np.full(nLoci, 9, dtype = np.int8)

        for i in range(nLoci):
            geno = raw_genotypes[i]
            if geno == 0:
                pat_hap[i] = 0
                mat_hap[i] = 0
            if geno == 1:
                pat_hap[i] = 0
                mat_hap[i] = 1
            if geno == 2:
                pat_hap[i] = 1
                mat_hap[i] = 0
            if geno == 3:
                pat_hap[i] = 1
                mat_hap[i] = 1
        return pat_hap, mat_hap


    # @jit(nopython=True, nogil=True) 
    def haplib_sample(self, bw_library, ind, rate, error_rate):
        hap_info = HaplotypeInformation(bw_library)
        nHaps, nLoci = bw_library.a.shape


        self.forward = ForwardHaplotype(nLoci, bw_library.full_nLoci)

        current_state = ((0, nHaps), (0, nHaps))
      
        genotypes = np.full(nLoci, 9, dtype = np.int64)
        rec = np.full(nLoci, 0, dtype = np.float32)
        values = np.full((4,4), 1, dtype = np.float32) # Just create this once.

        for i in range(nLoci):
            new_state, geno, rec[i] = self.sample_locus(current_state, i, bw_library, ind, values, rate, error_rate, hap_info)
            genotypes[i] = geno
            current_state = new_state

            self.forward.pat_ranges[0, i] = current_state[0][0]
            self.forward.pat_ranges[1, i] = current_state[0][1]

            self.forward.mat_ranges[0, i] = current_state[1][0]
            self.forward.mat_ranges[1, i] = current_state[1][1]


        # Add the final set of states
        hap_info.add_pat_sample(nLoci-1, new_state[0])
        hap_info.add_mat_sample(nLoci-1, new_state[1])
       
        return genotypes, rec, hap_info


    # @jit(nopython=True, nogil=True) 
    def sample_locus(self, current_states, index, bw_library, ind, values, rec_rate, error_rate, hap_info):

        # We want to do a couple of things. 
        # Of the current states, we want to work out which combinations will be produced at the next loci.
        # then work out probabilities.
        true_index = bw_library.get_true_index(index)
        nHaps, nLoci = bw_library.a.shape

        if index != 0:
            current_pat = bw_library.update_state(current_states[0], index)
            current_mat = bw_library.update_state(current_states[1], index)
        else:
            current_pat = bw_library.get_null_state(index)
            current_mat = bw_library.get_null_state(index)
        hap_lib = bw_library.get_null_state(index)

        exclusion = np.empty(0, dtype = np.int64)
        if ind.has_own_haplotypes:
            exclusion = ind.own_haplotypes[:, true_index]

        current_pat_counts = (count_haps(current_pat[0], exclusion), count_haps(current_pat[1], exclusion))
        current_mat_counts = (count_haps(current_mat[0], exclusion), count_haps(current_mat[1], exclusion))
        
        hap_lib_counts = (count_haps(hap_lib[0], exclusion), count_haps(hap_lib[1], exclusion))
        

        # # Recombination ordering. Could do 2 x 2 I guess...
        # # nn, nr, rn, rr

        get_haps_probs(values[0,:], current_pat_counts, current_mat_counts,  (1-rec_rate)*(1-rec_rate))
        get_haps_probs(values[1,:], current_pat_counts, hap_lib_counts, (1-rec_rate)*rec_rate)  
        get_haps_probs(values[2,:], hap_lib_counts, current_mat_counts, rec_rate*(1-rec_rate))
        get_haps_probs(values[3,:], hap_lib_counts, hap_lib_counts, rec_rate*rec_rate) 

        calculate_forward_geno_probs(values, self.forward.forward_geno_probs[:,true_index])

        for i in range(4):
            for j in range(4):
                # This is the individual's genotype probabilities. 
                values[i,j] *= ind.penetrance[j,true_index] * ind.backward[j, true_index]


        # Zero index of new_value is recombination status, one index is resulting genotype.
        
        new_value, value_sum = weighted_sample_2D(values)

        score = 0

        match_score = np.log(1-error_rate)
        no_match_score = np.log(error_rate)
        
        observed_genotype = ind.genotypes[true_index]
        selected_genotype = new_value[1]
        if observed_genotype != 9:
            if observed_genotype == 0:
                if selected_genotype == 0:
                    score -= match_score
                else:
                    score -= no_match_score

            if observed_genotype == 1:
                if selected_genotype == 1 or selected_genotype == 2:
                    score -= match_score
                else:
                    score -= no_match_score

            if observed_genotype == 2:
                if selected_genotype == 3:
                    score -= match_score
                else:
                    score -= no_match_score


        if value_sum == 0:
            new_state = current_states
            geno = np.argmax(ind.penetrance[:,true_index])
            score -= 2*np.log(1-rec_rate) # We search for lowest score
        else:

            geno = new_value[1]
            pat_value, mat_value = decode_genotype(new_value[1]) # Split out the genotype value into pat/mat states

            if new_value[0] == 0 or new_value[0] == 1:
                # No paternal recombination.
                pat_haps = current_pat[pat_value]
            else:
                # Paternal recombination
                pat_haps = hap_lib[pat_value]
                if index > 0: 
                    hap_info.add_pat_sample(index-1, current_states[0])

            if new_value[0] == 0 or new_value[0] == 2:
                # No maternal recombination.
                mat_haps = current_mat[mat_value]
            else:
                # maternal recombination
                mat_haps = hap_lib[mat_value]
                if index > 0:
                    hap_info.add_mat_sample(index-1, current_states[1])

           
            if new_value[0] == 1 or new_value[0] == 2:
                score -= np.log(rec_rate) + np.log(1-rec_rate) # We search for lowest score
            
            if new_value[0] == 3:
                score -= 2*np.log(rec_rate) # We search for lowest score

            new_state = (pat_haps, mat_haps)

        # print(self.ind.idx, new_state)

        return new_state, geno, score        


@jit(nopython=True, nogil=True)
def calculate_forward_geno_probs(geno_matrix, output):
    # Row sum and then normalize
    for i in range(4):
        output[i] = 0.00000001
        for j in range(4):
            output[i] += geno_matrix[j, i] # Second value is recombination state.

    norm_1D(output)



@jit(nopython=True, nogil = True)
def norm_1D(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total

@jit(nopython=True, nogil=True) 
def get_haplotypes(raw_genotypes):
    nLoci = len(raw_genotypes)
    pat_hap = np.full(nLoci, 9, dtype = np.int8)
    mat_hap = np.full(nLoci, 9, dtype = np.int8)

    for i in range(nLoci):
        geno = raw_genotypes[i]
        if geno == 0:
            pat_hap[i] = 0
            mat_hap[i] = 0
        if geno == 1:
            pat_hap[i] = 0
            mat_hap[i] = 1
        if geno == 2:
            pat_hap[i] = 1
            mat_hap[i] = 0
        if geno == 3:
            pat_hap[i] = 1
            mat_hap[i] = 1
    return pat_hap, mat_hap


@jit(nopython = True)
def count_haps(haplotypes, exclusion):

    base_count = haplotypes[1] - haplotypes[0]
    if len(exclusion) > 0:
        for i in range(len(exclusion)):
            if exclusion[i] >= haplotypes[0] and exclusion[i] < haplotypes[1]:
                base_count -=1

    if base_count < 0:
        print(base_count, exclusion)
    return base_count

@jit(nopython=True, nogil=True) 
def get_haps_probs(values, pat_haps, mat_haps, scale):
    # scale accounts for recombination rates.

    if pat_haps[0] + pat_haps[1] > 0:
        prop_pat_0 = pat_haps[0]/(pat_haps[0] + pat_haps[1])
        prop_pat_1 = pat_haps[1]/(pat_haps[0] + pat_haps[1])
    else:
        prop_pat_0 = 0
        prop_pat_1 = 0


    if mat_haps[0] + mat_haps[1] > 0:
        prop_mat_0 = mat_haps[0]/(mat_haps[0] + mat_haps[1])
        prop_mat_1 = mat_haps[1]/(mat_haps[0] + mat_haps[1])
    else:
        prop_mat_0 = 0
        prop_mat_1 = 0

    values[0] = prop_pat_0 * prop_mat_0 * scale
    values[1] = prop_pat_0 * prop_mat_1 * scale
    values[2] = prop_pat_1 * prop_mat_0 * scale
    values[3] = prop_pat_1 * prop_mat_1 * scale

    # return values


@jit(nopython=True, nogil=True) 
def weighted_sample_2D(mat):
    # Get sum of values    
    total = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            total += mat[i, j]
    value = random.random()*total

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value -= mat[i,j]
            if value < 0:
                return (i, j), total


    return (0,0), total


@jit(nopython=True, nogil=True) 
def weighted_sample_1D(mat):
    # Get sum of values    
    total = 0
    for i in range(mat.shape[0]):
        total += mat[i]
    value = random.random()*total

    # Select value
    for i in range(mat.shape[0]):
        value -= mat[i]
        if value < 0:
            return i

    return -1


@jit(nopython=True, nogil=True) 
def decode_genotype(geno):
    if geno == 0:
        return (0,0)
    if geno == 1:
        return (0, 1)
    if geno == 2:
        return (1, 0)
    if geno == 3:
        return (1, 1)

    return (20, 20)


### 
### The following is a bunch of code to handle consensus of multiple samples.
### This should probably be condensed and made better.
###

spec = OrderedDict()

tmp = PhasingSample(0.01, 0.01)
spec['samples'] = numba.optional(numba.typeof([tmp, tmp]))
spec['bw_library'] = numba.typeof(BurrowsWheelerLibrary.get_example_library().library)
spec['ind'] = numba.typeof(ImputationIndividual.get_example_phasing_individual())

@jitclass(spec)
class PhasingSampleContainer(object):
    def __init__(self, bw_library, ind):
        self.samples = None
        self.bw_library = bw_library
        self.ind = ind

    def add_sample(self, rate, error_rate):
        new_sample = PhasingSample(rate, error_rate)
        new_sample.sample(self.bw_library, self.ind)
        if self.samples is None:
            self.samples = [new_sample]
        else:
            self.samples += [new_sample]

    def get_consensus(self, sample_size):
        if len(self.samples) == 1:
            return self.samples[0].haplotypes

        nHaps = len(self.samples)
        nLoci = len(self.samples[0].genotypes)

        haplotypes = np.full((nHaps, 2, nLoci), 0,  dtype = np.int64)

        for i in range(nHaps):
            for j in range(2):
                haplotypes[i, j, :] = self.samples[i].haplotypes[j]


        rec_scores = np.full((nHaps, nLoci), 0,  dtype = np.int64)
        for i in range(nHaps):
            rec_scores[i, :] = count_regional_rec(self.samples[i].rec, sample_size)

        # genotypes = self.get_consensus_genotypes(haplotypes)
        genotypes = self.get_consensus_genotypes_smallest_region_rec(haplotypes, rec_scores)
        return self.get_consensus_haplotype(haplotypes, genotypes)

    def get_consensus_haplotype(self, haplotypes, genotypes):
        nHaps, tmp, nLoci = haplotypes.shape
        alignment = np.full(nHaps, 0, dtype = np.int8)
        
        haps = (np.full(nLoci, 9, dtype = np.int8), np.full(nLoci, 9, dtype = np.int8))

        for i in range(nLoci):
            if genotypes[i] == 0:
                haps[0][i] = 0
                haps[1][i] = 0
            if genotypes[i] == 2:
                haps[0][i] = 1
                haps[1][i] = 1

            if genotypes[i] == 1:
                count0 = 0
                count1 = 0

                for j in range(nHaps):
                    geno = haplotypes[j, 0, i] + haplotypes[j, 1, i] 
                    if geno == 1:
                        # If the genotype is not 1, throw out the haplotype for this loci.
                        if haplotypes[j, alignment[j], i] == 0:
                            count0 += 1
                        else:
                            count1 += 1

                # Set the haplotype
                if count0 >= count1:
                    haps[0][i] = 0
                    haps[1][i] = 1
                else:
                    haps[0][i] = 1
                    haps[1][i] = 0

                # Set alignment:
                for j in range(nHaps):
                    geno = haplotypes[j, 0, i] + haplotypes[j, 1, i] 
                    if geno == 1:
                        if haplotypes[j, 0, i] == haps[0][i]:
                            alignment[j] = 0
                        else:
                            alignment[j] = 1

        return haps

    def get_consensus_genotypes(self, haplotypes):
        nHaps, tmp, nLoci = haplotypes.shape
        genotypes = np.full(nLoci, 0, dtype = np.int8)
        p = np.full(3, 0, dtype = np.int32)
        for i in range(nLoci):
            p[:] = 0
            for j in range(nHaps):
                geno = haplotypes[j, 0, i] + haplotypes[j, 1, i]      
                p[geno] += 1

            genotypes[i] = get_max_index(p)

        return genotypes


    def get_consensus_genotypes_max_path_length(self, haplotypes, rec_scores):
        nHaps, tmp, nLoci = haplotypes.shape

        genotypes = np.full(nLoci, 0, dtype = np.int8)
        for i in range(nLoci):
            
            score = 0    
            index = 0
            for j in range(nHaps):
                if rec_scores[j, i] > score:
                    score = rec_scores[j, i]
                    index = j
            genotypes[i] = haplotypes[index, 0, i] + haplotypes[index, 1, i]      

        return genotypes


    def get_consensus_genotypes_smallest_region_rec(self, haplotypes, rec_scores):
        nHaps, tmp, nLoci = haplotypes.shape

        genotypes = np.full(nLoci, 0, dtype = np.int8)
        
        p = np.full(3, 0, dtype = np.int32)

        for i in range(nLoci):
            
            score = nLoci    
            index = 0
            for j in range(nHaps):
                if rec_scores[j, i]< score:
                    score = rec_scores[j, i]
            p[:] = 0
            count = 0
            for j in range(nHaps):
                if rec_scores[j, i] == score:
                    geno = haplotypes[j, 0, i] + haplotypes[j, 1, i]      
                    p[geno] += 1
                    count +=1
            genotypes[i] = get_max_index(p)

        return genotypes


# @jit(nopython=True, nogil=True) 
# def calculate_rec_distance(rec):
#     nLoci = len(rec)
#     forward = np.full(nLoci, 0, dtype = np.int64)
#     backward = np.full(nLoci, 0, dtype = np.int64)
    
#     count = nLoci + 1
#     for i in range(nLoci):
#         count += 1
#         if rec[i] >= 1:
#             count = 0
#         forward[i] = count

#     count = nLoci + 1
#     for i in range(nLoci-1, -1, -1):
#         count += 1
#         if rec[i] >= 1:
#             count = 0
#         backward[i] = count

#     combined = np.full(nLoci, 0, dtype = np.int64)
#     for i in range(nLoci):
#         combined[i] = min(forward[i], backward[i])

#     return combined


@jit(nopython=True, nogil=True) 
def count_regional_rec(rec, region = 25):
    nLoci = len(rec)
    forward = np.full(nLoci, 0, dtype = np.float32)
 
    count = nLoci + 1
    for i in range(nLoci):
        count += rec[i]
        forward[i] = count

    combined = np.full(nLoci, 0, dtype = np.float32)
    for i in range(nLoci):
        start = max(0, i - region)
        end = start + region*2
        if end >= nLoci:
            end = nLoci-1
            start = end - region*2

        combined[i] = forward[end] - forward[start]

    return combined



@jit(nopython=True, nogil=True) 
def get_max_index(array) :
    max_index = 0
    max_value = array[0]
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_index = i
            max_value = array[i]
    return max_index
