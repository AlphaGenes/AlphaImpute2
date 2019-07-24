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
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0)

    for rep in range(5):
        phase_round(individuals, individual_exclusion = True, set_haplotypes = False)
        
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

@jit(nopython=True, nogil=True) 
def phase_group(individuals, haplotype_library, set_haplotypes):
    for ind in individuals:
        phase(ind, haplotype_library, set_haplotypes = set_haplotypes)


@jit(nopython=True, nogil=True) 
def phase(ind, haplotype_library, set_haplotypes = False) :
    nLoci = len(ind.genotypes)
    rate = 5/nLoci

    if set_haplotypes:
        n_samples = 40
    else:
        n_samples = 1

    samples = PhasingSampleContainer(haplotype_library, ind)
    for i in range(n_samples):
        samples.add_sample(rate)

    pat_hap, mat_hap = samples.get_consensus(400)

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


###
### Actual sampler object 
###

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
        encoding_loci = index # Encoded at the index.
        new_range = HaplotypeRange(start, stop, hap_range, encoding_loci)  

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
        encoding_loci = index # Encoded at the index.
        new_range = HaplotypeRange(start, stop, hap_range, encoding_loci)  

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


    def convert_to_global_matrix(self, hap):
        if hap == 0:
            ranges = self.pat_ranges
        if hap == 1:
            ranges = self.mat_ranges

        range_index = np.full(self.bw_library.full_nLoci, -1, dtype = np.int64)
        residual_length = np.full(self.bw_library.full_nLoci, -1, dtype = np.int64)
        




spec = OrderedDict()
spec['genotypes'] = numba.int8[:]
spec['rec'] = numba.int64[:]

spec['rate'] = numba.float32
spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

tmp_info = HaplotypeInformation(BurrowsWheelerLibrary.get_example_library().library)
spec['hap_info'] = numba.typeof(tmp_info)

@jitclass(spec)
class PhasingSample(object):

    def __init__(self, rate):
        self.rate = rate

    def sample(self, bw_library, ind):
        raw_genotypes, rec, self.hap_info= haplib_sample(bw_library, ind, self.rate)
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

@jit(nopython=True, nogil=True) 
def haplib_sample(bw_library, ind, rate):
    hap_info = HaplotypeInformation(bw_library)

    nHaps, nLoci = bw_library.a.shape

    current_state = ((0, nHaps), (0, nHaps))
  
    genotypes = np.full(nLoci, 9, dtype = np.int64)
    rec = np.full(nLoci, 0, dtype = np.int64)
    values = np.full((4,4), 1, dtype = np.float32) # Just create this once.

    for i in range(nLoci):
        new_state, geno, rec[i] = sample_locus(current_state, i, bw_library, ind, values, rate, hap_info)
        genotypes[i] = geno
        current_state = new_state

    # Add the final set of states
    hap_info.add_pat_sample(nLoci-1, new_state[0])
    hap_info.add_mat_sample(nLoci-1, new_state[1])

    return genotypes, rec, hap_info


@jit(nopython=True, nogil=True) 
def sample_locus(current_states, index, bw_library, ind, values, rate, hap_info):

    # We want to do a couple of things. 
    # Of the current states, we want to work out which combinations will be produced at the next loci.
    # then work out probabilities.
    true_index = bw_library.get_true_index(index)
    nHaps, nLoci = bw_library.a.shape

    rec = rate

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

    get_haps_probs(values[0,:], current_pat_counts, current_mat_counts,  (1-rec)*(1-rec))
    get_haps_probs(values[1,:], current_pat_counts, hap_lib_counts, (1-rec)*rec)  
    get_haps_probs(values[2,:], hap_lib_counts, current_mat_counts, rec*(1-rec))
    get_haps_probs(values[3,:], hap_lib_counts, hap_lib_counts, rec*rec) 

    for i in range(4):
        for j in range(4):
            # This is the individual's genotype probabilities. 
            values[i,j] *= ind.penetrance[j,true_index]


    # Zero index of new_value is recombination status, one index is resulting genotype.
    
    new_value = weighted_sample_2D(values)

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

    rec = 0
    if new_value[0] == 1 or new_value[0] == 2:
        rec = 1
    if new_value[0] == 3:
        rec = 2

    new_state = (pat_haps, mat_haps)

    # print(self.ind.idx, new_state)

    return new_state, geno, rec        

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
    # total = np.sum(mat)
    
    total = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            total += mat[i, j]
    value = random.random()*total

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value -= mat[i,j]
            if value < 0:
                return (i, j)


    return (0,0)


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

tmp = PhasingSample(0.01)
spec['samples'] = numba.optional(numba.typeof([tmp, tmp]))
spec['bw_library'] = numba.typeof(BurrowsWheelerLibrary.get_example_library().library)
spec['ind'] = numba.typeof(ImputationIndividual.get_example_phasing_individual())

@jitclass(spec)
class PhasingSampleContainer(object):
    def __init__(self, bw_library, ind):
        self.samples = None
        self.bw_library = bw_library
        self.ind = ind

    def add_sample(self, rate):
        new_sample = PhasingSample(rate)
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


@jit(nopython=True, nogil=True) 
def calculate_rec_distance(rec):
    nLoci = len(rec)
    forward = np.full(nLoci, 0, dtype = np.int64)
    backward = np.full(nLoci, 0, dtype = np.int64)
    
    count = nLoci + 1
    for i in range(nLoci):
        count += 1
        if rec[i] >= 1:
            count = 0
        forward[i] = count

    count = nLoci + 1
    for i in range(nLoci-1, -1, -1):
        count += 1
        if rec[i] >= 1:
            count = 0
        backward[i] = count

    combined = np.full(nLoci, 0, dtype = np.int64)
    for i in range(nLoci):
        combined[i] = min(forward[i], backward[i])

    return combined


@jit(nopython=True, nogil=True) 
def count_regional_rec(rec, region = 25):
    nLoci = len(rec)
    forward = np.full(nLoci, 0, dtype = np.int64)
 
    count = nLoci + 1
    for i in range(nLoci):
        count += rec[i]
        forward[i] = count

    combined = np.full(nLoci, 0, dtype = np.int64)
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
