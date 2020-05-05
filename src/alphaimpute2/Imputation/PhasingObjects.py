import numba
import numpy as np
import random

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass


from numba import njit, jit
from collections import OrderedDict

# numba.NUMBA_DEBUGINFO=1

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
            self.mat_ranges.append(new_range)

        
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
            self.pat_ranges.append(new_range)

    def get_global_bounds(self, index, hap):
        if hap == 0:
            ranges = self.pat_ranges
        if hap == 1:
            ranges = self.mat_ranges

        # The 1 offset is because sub_start should be included in the previous range (and not the current range), and then we include sub_end in our current range.
        global_start = self.bw_library.get_true_index(ranges[index].start)+1

        # The 1 offset is because true_end should be included in the current range, and this is a python range so we need to go 1 further.
        global_end = self.bw_library.get_true_index(ranges[index].stop)+1

        # Over-ride the start and stop bounds for the first and last haplotypes.
        if index == 0:
            global_start = 0
        if index == len(ranges) - 1:
            global_end = self.bw_library.full_nLoci

        return global_start, global_end

spec = OrderedDict()

# Potentially, I think we can kill this structure and re-integrate it with the sample.
# We could also remove the above haplotype structure in favor of the per-locus values.
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

spec['rec_rate'] = numba.float32
spec['error_rate'] = numba.float32

spec['match_score'] = numba.float32
spec['no_match_score'] = numba.float32
spec['rec_score'] = numba.float32
spec['no_rec_score'] = numba.float32

spec['haplotypes'] = numba.typeof((np.array([0, 1], dtype = np.int8), np.array([0], dtype = np.int8)))

tmp_info = HaplotypeInformation(BurrowsWheelerLibrary.get_example_library().library)
spec['hap_info'] = numba.typeof(tmp_info)

tmp_forward = ForwardHaplotype(10, 100)
spec['forward'] = numba.typeof(tmp_forward)

spec['calculate_forward_estimates'] = numba.boolean
spec['track_hap_info'] = numba.boolean

@jitclass(spec)
class PhasingSample(object):
    def __init__(self, rec_rate, error_rate):
        self.rec_rate = rec_rate
        self.error_rate = error_rate
        self.calculate_forward_estimates = True
        self.track_hap_info = True

        self.match_score = np.log(1-error_rate)
        self.no_match_score = np.log(error_rate)

        self.rec_score = np.log(rec_rate)
        self.no_rec_score = np.log(1-rec_rate)


    def sample(self, bw_library, ind):
        # raw_genotypes = haplib_sample_alt(self, bw_library, ind)
        raw_genotypes = haplib_sample(self, bw_library, ind)
        self.haplotypes = get_haplotypes(raw_genotypes)
        self.genotypes = self.haplotypes[0] + self.haplotypes[1]

@jit(nopython = True, nogil=True)
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



@jit(nopython = True, nogil=True)
def haplib_sample(sample, bw_library, ind):
    #### So this is going to get sorta ugly pretty fast.

    nHaps, nLoci = bw_library.a.shape

    sample.hap_info = HaplotypeInformation(bw_library)
    sample.forward = ForwardHaplotype(nLoci, bw_library.full_nLoci)
    sample.rec = np.empty(nLoci, dtype = np.float32)

    previous_state = ((0, nHaps), (0, nHaps))
  
    genotypes = np.empty(nLoci, dtype = np.int8)
    rec_states = np.empty(nLoci, dtype = np.int8)

    geno_probs = np.full((4,4), 1, dtype = np.float32) # Just create this once.

    ### Local variables.
    bw_loci = bw_library.loci
    zeroOccNext = bw_library.zeroOccNext
    nZeros = bw_library.nZeros
    nHaps, nLoci = zeroOccNext.shape

    own_haplotypes = ind.own_haplotypes
    has_own_haplotypes = ind.has_own_haplotypes

    rec_rate = sample.rec_rate # Constant.
    calculate_forward_estimates = sample.calculate_forward_estimates

    forward_geno_probs = sample.forward.forward_geno_probs
    penetrance_and_backward = ind.penetrance*ind.backward
    penetrance_and_backward = penetrance_and_backward/np.sum(penetrance_and_backward, axis = 0)

    ind_genotypes = ind.genotypes

    match_score = sample.match_score
    no_match_score = sample.no_match_score

    rec_score = sample.rec_score
    no_rec_score = sample.no_rec_score
 
    rec = sample.rec

    pat_ranges = sample.forward.pat_ranges
    mat_ranges = sample.forward.mat_ranges


    for index in range(nLoci):

        true_index = bw_loci[index]
        # Overall sampling pipeline:
    
        # STATE UPDATE:
        # Take the current set of states and update them for the next loci.
        # This will split the states out into the states that are 0 or 1.
        
        # GENOTYPE ESTIMATES:
        # Use the updated state information to estimate the genotypes.
        # Figure out the proportion of states that transalte to either 0 or 1.
        # Then construct a 4x4 matrix looking at combinations of recombinations and 

        # SAMPLE AND PROCESS:
        # Use the estimated geno_probs matrix to sample a genotype state and a recombination state.
        # Update the current states based on the genotype + recombination choice.
        # Create a score that includes the genotype log-likelihood and the recombination log-likelihood.


        # Get new states. We will need these later for selecting a state update.
        current_pat, current_mat, hap_lib = update_states(previous_state, index, nZeros, zeroOccNext)

        # Calculate genotype propotions for the next state, potentially excluding an individual's own haplotype.
        exclusion = (own_haplotypes[0, true_index], own_haplotypes[1, true_index]) 
        pat_prop, mat_prop, hap_lib_prop = get_proportions(current_pat, current_mat, hap_lib, has_own_haplotypes, exclusion)

        # Calculate genotype probabilities. 
        calculate_haps_probs(rec_rate, geno_probs, pat_prop, mat_prop, hap_lib_prop)

        if calculate_forward_estimates:
            calculate_forward_geno_probs(geno_probs, forward_geno_probs[:,true_index])

        add_penetrance(geno_probs, penetrance_and_backward[:,true_index])

        # Select a new sample. 
        rec_state, selected_genotype = weighted_sample_2D(geno_probs)


        # Calculate the score for the locus.
        score = 0
        observed_genotype = ind_genotypes[true_index]
        score += score_from_genotype(observed_genotype, selected_genotype, match_score, no_match_score)
        # score += score_from_geno_probs(penetrance_and_backward[:,true_index], selected_genotype) # This doesn't seem to help much in the limited test case.
        score += score_from_rec_state(rec_state, rec_score, no_rec_score)
        rec[index] = score

        # get the next set of states.
        current_state = get_new_state(selected_genotype, rec_state, current_pat, current_mat, hap_lib)


        # Assign variables for the next iteration.
        genotypes[index] = selected_genotype
        rec_states[index] = rec_state
        pat_ranges[0, index] = current_state[0][0]
        pat_ranges[1, index] = current_state[0][1]

        mat_ranges[0, index] = current_state[1][0]
        mat_ranges[1, index] = current_state[1][1]

    
        # if rec_state > 0 and self.track_hap_info: 
        #     self.update_hap_info(index, rec_state, previous_states)

        previous_state = current_state
    
    sample.forward.forward_geno_probs = forward_geno_probs

    # Add the final set of states
    track_hap_info = sample.track_hap_info
    if track_hap_info:
        for index in range(nLoci):          
            if index > 0 and rec_states[index] > 0 and track_hap_info: 

                previous_state = ((pat_ranges[0, index -1], pat_ranges[1, index -1]),(mat_ranges[0, index -1], mat_ranges[1, index -1]))
                update_hap_info(sample.hap_info, index, rec_states[index], previous_state)

        sample.hap_info.add_pat_sample(nLoci-1, current_state[0])
        sample.hap_info.add_mat_sample(nLoci-1, current_state[1])
   
    return genotypes





@jit(nopython = True, nogil=True)
def haplib_sample_alt(sample, bw_library, ind):
    #### So this is going to get sorta ugly pretty fast.

    nHaps, nLoci = bw_library.a.shape

    sample.hap_info = HaplotypeInformation(bw_library)
    sample.forward = ForwardHaplotype(nLoci, bw_library.full_nLoci)
    sample.rec = np.empty(nLoci, dtype = np.float32)

    previous_state = ((0, nHaps), (0, nHaps))
  
    genotypes = np.empty(nLoci, dtype = np.int8)
    rec_states = np.empty(nLoci, dtype = np.int8)

    geno_probs = np.full((4,4), 1, dtype = np.float32) # Just create this once.

    ### Local variables.
    bw_loci = bw_library.loci
    zeroOccNext = bw_library.zeroOccNext
    nZeros = bw_library.nZeros
    nHaps, nLoci = zeroOccNext.shape

    own_haplotypes = ind.own_haplotypes
    has_own_haplotypes = ind.has_own_haplotypes

    rec_rate = sample.rec_rate # Constant.
    calculate_forward_estimates = sample.calculate_forward_estimates

    forward_geno_probs = sample.forward.forward_geno_probs
    penetrance_and_backward = ind.penetrance*ind.backward
    penetrance_and_backward = penetrance_and_backward/np.sum(penetrance_and_backward, axis = 0)

    ind_genotypes = ind.genotypes

    match_score = sample.match_score
    no_match_score = sample.no_match_score

    rec_score = sample.rec_score
    no_rec_score = sample.no_rec_score
 
    rec = sample.rec

    pat_ranges = sample.forward.pat_ranges
    mat_ranges = sample.forward.mat_ranges




    for index in range(nLoci):

        true_index = bw_loci[index]
        # Overall sampling pipeline:
    
        # STATE UPDATE:
        # Take the current set of states and update them for the next loci.
        # This will split the states out into the states that are 0 or 1.
        
        # GENOTYPE ESTIMATES:
        # Use the updated state information to estimate the genotypes.
        # Figure out the proportion of states that transalte to either 0 or 1.
        # Then construct a 4x4 matrix looking at combinations of recombinations and 

        # SAMPLE AND PROCESS:
        # Use the estimated geno_probs matrix to sample a genotype state and a recombination state.
        # Update the current states based on the genotype + recombination choice.
        # Create a score that includes the genotype log-likelihood and the recombination log-likelihood.


        # Get new states. We will need these later for selecting a state update.

        #ORIGINAL
        # current_pat, current_mat, hap_lib = update_states(previous_state, index, nZeros, zeroOccNext)

        hap_lib = BurrowsWheelerLibrary.get_null_state(index, nZeros, zeroOccNext)

        if index != 0:
            current_pat = BurrowsWheelerLibrary.update_state(current_states[0], index, nZeros, zeroOccNext)
            current_mat = BurrowsWheelerLibrary.update_state(current_states[1], index, nZeros, zeroOccNext)
        else:
            current_pat = hap_lib
            current_mat = hap_lib
        

        # Calculate genotype propotions for the next state, potentially excluding an individual's own haplotype.
        exclusion = (own_haplotypes[0, true_index], own_haplotypes[1, true_index]) 
        pat_prop, mat_prop, hap_lib_prop = get_proportions(current_pat, current_mat, hap_lib, has_own_haplotypes, exclusion)

        # Calculate genotype probabilities. 

        # ORIGINAL
        # calculate_haps_probs(rec_rate, geno_probs, pat_prop, mat_prop, hap_lib_prop)

        #NEW START
        for i in range(4):
            if i == 0:
                tmp_pat_prob = pat_prop
                tmp_mat_prob = mat_prop
                scale = (1-rec_rate)*(1-rec_rate)
            if i == 1:
                tmp_pat_prob = pat_prop
                tmp_mat_prob = hap_lib_prop
                scale = (1-rec_rate)*rec_rate
            if i == 2:
                tmp_pat_prob = hap_lib_prop
                tmp_mat_prob = mat_prop
                scale = (1-rec_rate)*rec_rate
            if i == 3:
                tmp_pat_prob = hap_lib_prop
                tmp_mat_prob = hap_lib_prop
                scale = rec_rate*rec_rate

            values[i, 0] = tmp_pat_prob[0] * tmp_mat_prop[0] * scale
            values[i, 1] = tmp_pat_prob[0] * tmp_mat_prop[1] * scale
            values[i, 2] = tmp_pat_prob[1] * tmp_mat_prop[0] * scale
            values[i, 3] = tmp_pat_prob[1] * tmp_mat_prop[1] * scale
        #NEW END


        # if calculate_forward_estimates:
        #     calculate_forward_geno_probs(geno_probs, forward_geno_probs[:,true_index])

        # ORIGINAL LINE
        # add_penetrance(geno_probs, penetrance_and_backward[:,true_index])

        for j in range(4):
            for i in range(4):
                # This is the individual's genotype probabilities. 
                geno_probs[i,j] *= penetrance_and_backward[j, true_index]


        # ORIGINAL LINE
        # Select a new sample. 
        # rec_state, selected_genotype = weighted_sample_2D(geno_probs)
        stop = False
        total = 0
        mat = geno_probs
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                total += mat[i, j]
        if total == 0:
            stop = True
            rec_state, selected_genotype = (-1, -1)
        if not stop:
            value = random.random()*total

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    value -= mat[i,j]
                    if value < 0:
                        rec_state, selected_genotype =  (i, j)
                        stop = True
                        break

        if not stop:
            rec_state, selected_genotype = (0,0)


        # Calculate the score for the locus.
        score = 0
        observed_genotype = ind_genotypes[true_index]

        # ORIGINAL LINE:
        # score += score_from_genotype(observed_genotype, selected_genotype, match_score, no_match_score)
        new_score = 0
        if observed_genotype != 9:
            error = True
            if observed_genotype == 0 and selected_genotype == 0:
                error = False
            elif observed_genotype == 1 and (selected_genotype == 1 or selected_genotype == 2):
                error = False
            elif observed_genotype == 2 and selected_genotype == 3:
                error = False

            if error:
                new_score = no_match_score
            else:
                new_score = match_score

        score += -new_score


        # ORIGINAL LINE:
        # score += score_from_rec_state(rec_state, rec_score, no_rec_score)
        new_score = 0
        if rec_state == 0:
            new_score = 2*no_rec_score 

        if rec_state == 1 or rec_state == 2:
            new_score = rec_score + no_rec_score # We search for lowest score

        if rec_state == 3:
            new_score = 2*rec_score 

        score += -new_score

        rec[index] = score

        # get the next set of states.

        # ORIGINAL LINE
        # current_state = get_new_state(selected_genotype, rec_state, current_pat, current_mat, hap_lib)

        if selected_genotype == 0:
            pat_value, mat_value = (0,0)
        elif selected_genotype == 1:
            pat_value, mat_value = (0, 1)
        elif selected_genotype == 2:
            pat_value, mat_value = (1, 0)
        elif selected_genotype == 3:
            pat_value, mat_value = (1, 1)

        if rec_state == 0 or rec_state == 1:
            # No paternal recombination.
            pat_haps = current_pat[pat_value]
        else:
            # Paternal recombination
            pat_haps = hap_lib[pat_value]

        if rec_state == 0 or rec_state == 2:
            # No maternal recombination.
            mat_haps = current_mat[mat_value]
        else:
            # maternal recombination
            mat_haps = hap_lib[mat_value]

        current_state = (pat_haps, mat_haps)



        # Assign variables for the next iteration.
        genotypes[index] = selected_genotype
        rec_states[index] = rec_state
        pat_ranges[0, index] = current_state[0][0]
        pat_ranges[1, index] = current_state[0][1]

        mat_ranges[0, index] = current_state[1][0]
        mat_ranges[1, index] = current_state[1][1]

    
        previous_state = current_state
    
    sample.forward.forward_geno_probs = forward_geno_probs

    # Add the final set of states
    track_hap_info = sample.track_hap_info
    if track_hap_info:
        for index in range(nLoci):          
            if index > 0 and rec_states[index] > 0 and track_hap_info: 

                previous_state = ((pat_ranges[0, index -1], pat_ranges[1, index -1]),(mat_ranges[0, index -1], mat_ranges[1, index -1]))
                update_hap_info(sample.hap_info, index, rec_states[index], previous_state)

        sample.hap_info.add_pat_sample(nLoci-1, current_state[0])
        sample.hap_info.add_mat_sample(nLoci-1, current_state[1])
   
    return genotypes


@jit(nopython = True, nogil=True)
def add_penetrance(geno_probs, penetrance_and_backward):
    for j in range(4):
        for i in range(4):
            # This is the individual's genotype probabilities. 
            geno_probs[i,j] *= penetrance_and_backward[j]



@jit(nopython = True, nogil=True)
def get_proportions(current_pat, current_mat, hap_lib, has_own_haplotypes, exclusion):

    if has_own_haplotypes:
        # exclusion = (ind.own_haplotypes[0, true_index], ind.own_haplotypes[1, true_index]) 

        pat_counts = (count_haps_with_exclusion(current_pat[0], exclusion), count_haps_with_exclusion(current_pat[1], exclusion))
        mat_counts = (count_haps_with_exclusion(current_mat[0], exclusion), count_haps_with_exclusion(current_mat[1], exclusion))
        hap_lib_counts = (count_haps_with_exclusion(hap_lib[0], exclusion), count_haps_with_exclusion(hap_lib[1], exclusion))

    else:
        pat_counts = (count_haps(current_pat[0]), count_haps(current_pat[1]))
        mat_counts = (count_haps(current_mat[0]), count_haps(current_mat[1]))            
        hap_lib_counts = (count_haps(hap_lib[0]), count_haps(hap_lib[1]))
    
    pat_prop = calculate_proportions(pat_counts)
    mat_prop = calculate_proportions(mat_counts)
    hap_lib_prop = calculate_proportions(hap_lib_counts)

    return pat_prop, mat_prop, hap_lib_prop






@jit(nopython = True, nogil=True)
def count_haps(haplotypes):
    return haplotypes[1] - haplotypes[0]

@jit(nopython = True, nogil=True)
def count_haps_with_exclusion(haplotypes, exclusion):

    base_count = haplotypes[1] - haplotypes[0]
    # Only two haplotypes to exclude. May need to expand later.
    if exclusion[0] >= haplotypes[0] and exclusion[0] < haplotypes[1]:
        base_count -=1

    if exclusion[1] >= haplotypes[0] and exclusion[1] < haplotypes[1]:
        base_count -=1

    return base_count


@jit(nopython = True, nogil=True)
def calculate_proportions(hap_counts):

    if hap_counts[0] + hap_counts[1] > 0:
        prop_0 = hap_counts[0]/(hap_counts[0] + hap_counts[1])
        prop_1 = hap_counts[1]/(hap_counts[0] + hap_counts[1])
    else:
        prop_0 = 0
        prop_1 = 0

    return (prop_0, prop_1)


@jit(nopython = True, nogil=True)
def calculate_haps_probs(rec_rate, values, pat_prop, mat_prop, hap_lib_prop):

    # # Recombination ordering:
    # # nn, nr, rn, rr

    fill_values(values[0,:], pat_prop, mat_prop, (1-rec_rate)*(1-rec_rate))
    fill_values(values[1,:], pat_prop, hap_lib_prop, (1-rec_rate)*rec_rate)  
    fill_values(values[2,:], hap_lib_prop, mat_prop, rec_rate*(1-rec_rate))
    fill_values(values[3,:], hap_lib_prop, hap_lib_prop, rec_rate*rec_rate) 


@jit(nopython = True, nogil=True)
def fill_values(sub_values, pat_prob, mat_prop, scale):
    sub_values[0] = pat_prob[0] * mat_prop[0] * scale
    sub_values[1] = pat_prob[0] * mat_prop[1] * scale
    sub_values[2] = pat_prob[1] * mat_prop[0] * scale
    sub_values[3] = pat_prob[1] * mat_prop[1] * scale


@jit(nopython = True, nogil=True)
def score_from_geno_probs(geno_probs, selected_genotype):
    return -np.log(geno_probs[selected_genotype])


@jit(nopython = True, nogil=True)
def score_from_genotype(observed_genotype, selected_genotype, match_score, no_match_score):
    score = 0

    if observed_genotype != 9:
        error = True
        if observed_genotype == 0 and selected_genotype == 0:
            error = False
        elif observed_genotype == 1 and (selected_genotype == 1 or selected_genotype == 2):
            error = False
        elif observed_genotype == 2 and selected_genotype == 3:
            error = False

        if error:
            score = no_match_score
        else:
            score = match_score

    return -score


@jit(nopython = True, nogil=True)
def score_from_rec_state(rec_state, rec_score, no_rec_score):

    if rec_state == 0:
        score = 2*no_rec_score 

    if rec_state == 1 or rec_state == 2:
        score = rec_score + no_rec_score # We search for lowest score

    if rec_state == 3:
        score = 2*rec_score 
    return -score


@jit(nopython = True, nogil=True)
def get_new_state(selected_genotype, rec_state, current_pat, current_mat, hap_lib):
    # Hap info is a sizeable amount of time.

    pat_value, mat_value = decode_genotype(selected_genotype) # Split out the genotype value into pat/mat states

    if rec_state == 0 or rec_state == 1:
        # No paternal recombination.
        pat_haps = current_pat[pat_value]
    else:
        # Paternal recombination
        pat_haps = hap_lib[pat_value]

    if rec_state == 0 or rec_state == 2:
        # No maternal recombination.
        mat_haps = current_mat[mat_value]
    else:
        # maternal recombination
        mat_haps = hap_lib[mat_value]

    new_state = (pat_haps, mat_haps)
    return new_state


@jit(nopython = True, nogil=True)
def update_hap_info(hap_info, index, rec_state, previous_states):
    if rec_state == 2 or rec_state == 3:
            hap_info.add_pat_sample(index-1, previous_states[0])

    if rec_state == 1 or rec_state == 3:
        hap_info.add_mat_sample(index-1, previous_states[1])


@jit(nopython = True, nogil=True)
def update_states(current_states, index, nZeros, zeroOccNext):
    hap_lib = BurrowsWheelerLibrary.get_null_state(index, nZeros, zeroOccNext)

    if index != 0:
        current_pat = BurrowsWheelerLibrary.update_state(current_states[0], index, nZeros, zeroOccNext)
        current_mat = BurrowsWheelerLibrary.update_state(current_states[1], index, nZeros, zeroOccNext)
    else:
        current_pat = hap_lib
        current_mat = hap_lib
    
    return current_pat, current_mat, hap_lib






#### END CLASS FUNCTIONS



@jit(nopython=True, nogil=True) 
def decode_genotype(geno):
    if geno == 0:
        return (0,0)
    elif geno == 1:
        return (0, 1)
    elif geno == 2:
        return (1, 0)
    elif geno == 3:
        return (1, 1)

    # No good reason to pick this, but wanted to pick something within the bounds.
    return (0,0)

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



@jit(nopython=True, nogil=True) 
def weighted_sample_2D(mat):
    # Draws a random index from mat, using the elements of mat as weights.

    total = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            total += mat[i, j]
    if total == 0:
        return (-1, -1)
    value = random.random()*total

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value -= mat[i,j]
            if value < 0:
                return (i, j)


    return (0,0)


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
    @profile
    def add_sample(self, rate, error_rate, calculate_forward_estimates, track_hap_info):
        new_sample = PhasingSample(rate, error_rate)
        new_sample.calculate_forward_estimates = calculate_forward_estimates
        new_sample.track_hap_info = track_hap_info
        
        new_sample.sample(self.bw_library, self.ind)
        if self.samples is None:
            self.samples = [new_sample]
        else:
            self.samples += [new_sample]

    @profile
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
        genotypes = get_consensus_genotypes_smallest_region_rec(haplotypes, rec_scores)
        return get_consensus_haplotype(haplotypes, genotypes)

@jit(nopython=True, nogil=True) 
def get_consensus_haplotype(haplotypes, genotypes):
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

@jit(nopython=True, nogil=True) 
def get_consensus_genotypes(haplotypes):
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


@jit(nopython=True, nogil=True) 
def get_consensus_genotypes_max_path_length(haplotypes, rec_scores):
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


@jit(nopython=True, nogil=True) 
def get_consensus_genotypes_smallest_region_rec(haplotypes, rec_scores):
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
