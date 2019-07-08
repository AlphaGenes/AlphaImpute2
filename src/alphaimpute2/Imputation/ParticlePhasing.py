import numba
from numba import njit, jit, int8, int32, boolean, jitclass
import numpy as np

from ..tinyhouse import InputOutput
from . import BurrowsWheelerLibrary

from . import Imputation
import random
import collections

import concurrent.futures
from itertools import repeat

np.seterr(all='raise')

try:
    profile
except:
    def profile(x): 
        return x

import datetime
def time_func(text):
    # This creates a decorator with "text" set to "text"
    def timer_dec(func):
        # This is the returned, modified, function
        def timer(*args, **kwargs):
            start_time = datetime.datetime.now()
            values = func(*args, **kwargs)
            print(text, (datetime.datetime.now() - start_time).total_seconds())
            return values
        return timer

    return timer_dec


@profile
def phase_individuals(individuals) :

    for ind in individuals:
        # We never call genotypes so can do this once.
        ind.peeling_view.setValueFromGenotypes(ind.phasing_view.penetrance, 0)


    for rep in range(5):
        phase_round(individuals, set_haplotypes = False)
    
    phase_round(individuals, set_haplotypes = True)


@time_func("Phasing round")
@profile
def phase_round(individuals, set_haplotypes = False):
    haplotype_library = HaplotypeLibrary()
    
    for ind in individuals:
        for hap in ind.phasing_view.current_haplotypes:
            # if np.mean(hap != 9) > .9:
            #     print(ind.idx)
            haplotype_library.append(hap.copy())
    
    haplotype_library.removeMissingValues()

    bwLibrary = createBWlibrary(haplotype_library)
    # bwLibrary = BurrowsWheelerLibrary.BurrowsWheelerLibrary(haplotype_library.library)

    for ind in individuals:
        phase(ind, bwLibrary, set_haplotypes = set_haplotypes)

@time_func("Creating BW library")
def createBWlibrary(haplotype_library):
    return BurrowsWheelerLibrary.BurrowsWheelerLibrary(haplotype_library.library)

@profile
def phase(ind, haplotype_library, set_haplotypes = False) :

    # pat_hap, mat_hap = sampler.get_haplotypes()
    if set_haplotypes:
        rate = 10
    else:
        rate = 10

    haplotypes = []
    for i in range(1):
        sampler = Sampler(ind, haplotype_library, rate = rate)
        sampler.sample()
        haplotypes.append(sampler.get_haplotypes())

    pat_hap, mat_hap = get_consensus(np.array(haplotypes))


    if set_haplotypes:
        ind.haplotypes[0][:] = pat_hap
        ind.haplotypes[1][:] = mat_hap
        ind.genotypes[:] = pat_hap + mat_hap
    else:
        ind.phasing_view.current_haplotypes[0][:] = pat_hap
        ind.phasing_view.current_haplotypes[1][:] = mat_hap
    


@profile
def get_consensus(haplotypes):
    genotypes = get_consensus_genotypes(haplotypes)
    return get_consensus_haplotype(haplotypes, genotypes)


@njit
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

# genotypes = np.array([1, 1, 1, 1, 1])
# haplotypes = np.array([
#    ((1, 0, 0, 0, 1),
#     (0, 1, 1, 1, 0)),
#    ((1, 0, 0, 0, 1),
#     (0, 1, 1, 1, 0)),
#    ((1, 0, 0, 1, 0),
#     (0, 1, 1, 0, 1))])

# get_consensus_haplotype(haplotypes, genotypes)

@njit
def get_consensus_genotypes(haplotypes):
    nHaps, tmp, nLoci = haplotypes.shape
    genotypes = np.full(nLoci, 0, dtype = np.int8)
    p = np.full(3, 0, dtype = np.int32)
    for i in range(nLoci):
        p[:] = 0
        for j in range(nHaps):
            geno = haplotypes[j, 0, i] + haplotypes[j, 1, i]      
            p[geno] += 1

        genotypes[i] = call(p)

    return genotypes

@njit
def call(array) :
    max_index = 0
    max_value = array[0]
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_index = i
            max_value = array[i]
    return max_index


class Sampler(object):

    def __init__(self, ind, haplotype_library, rate = 10):
        self.nLoci = len(ind.genotypes)
        self.ind = ind
        self.haplotype_library = haplotype_library

        self.genotypes = np.full(self.nLoci, 9, dtype = np.float32)

        self.rate = rate


    def get_haplotypes(self):
        # Converts genotypes to haplotypes.
        return self.get_haplotypes_numba(self.genotypes, self.nLoci)

    @staticmethod
    @njit
    def get_haplotypes_numba(genotypes, nLoci):
        pat_hap = np.full(nLoci, 9, dtype = np.int8)
        mat_hap = np.full(nLoci, 9, dtype = np.int8)

        for i in range(nLoci):
            geno = genotypes[i]
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

    def sample(self):
        self.genotypes = haplib_sample(self.haplotype_library.library, self.ind.phasing_view)

@jit(nopython=True)
def haplib_sample(haplotype_library, ind):
    nHaps, nLoci = haplotype_library.a.shape

    current_state = ((0, 1), (0, 1))

    # current_state = ([i for i in range(haplotype_library.nHaps)], [i for i in range(haplotype_library.nHaps)])
    
    genotypes = np.full(nLoci, 9, dtype = np.float32)
    values = np.full((4,4), 0, dtype = np.float32) # Just create this once.

    for i in range(nLoci):
        new_state, geno = sample_locus(current_state, i, haplotype_library, ind, values)
        genotypes[i] = geno
        current_state = new_state
    return genotypes

@jit(nopython=True)
def sample_locus(current_states, index, haplotype_library, ind, values):

    # We want to do a couple of things. 
    # Of the current states, we want to work out which combinations will be produced at the next loci.
    # then work out probabilities.
    nHaps, nLoci = haplotype_library.a.shape

    rec = 10/nLoci

    if index != 0:
        current_pat = haplotype_library.update_state(current_states[0], index)
        current_mat = haplotype_library.update_state(current_states[1], index)
    else:
        current_pat = haplotype_library.get_null_state(index)
        current_mat = haplotype_library.get_null_state(index)
    hap_lib = haplotype_library.get_null_state(index)



    current_pat_counts = (current_pat[0][1] - current_pat[0][0], current_pat[1][1] - current_pat[1][0])
    current_mat_counts = (current_mat[0][1] - current_mat[0][0], current_mat[1][1] - current_mat[1][0])
    
    hap_lib_counts = (hap_lib[0][1] - hap_lib[0][0], hap_lib[1][1] - hap_lib[1][0])


    # Recombination ordering. Could do 2 x 2 I guess...
    # nn, nr, rn, rr

    get_haps_probs(values[0,:], current_pat_counts, current_mat_counts,  (1-rec)*(1-rec))
    get_haps_probs(values[1,:], current_pat_counts, hap_lib_counts, (1-rec)*rec)  
    get_haps_probs(values[2,:], hap_lib_counts, current_mat_counts, rec*(1-rec))
    get_haps_probs(values[3,:], hap_lib_counts, hap_lib_counts, rec*rec) 

    for i in range(4):
        for j in range(4):
            # This is the individual's genotype probabilities. 
            values[i,j] *= ind.penetrance[j,index]


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

    if new_value[0] == 0 or new_value[0] == 2:
        # No maternal recombination.
        mat_haps = current_mat[mat_value]
    else:
        # maternal recombination
        mat_haps = hap_lib[mat_value]

    new_state = (pat_haps, mat_haps)

    # print(self.ind.idx, new_state)

    return new_state, geno        


@jit(nopython=True)
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


@njit
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


@njit
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

class HaplotypeLibrary(object) :
    def __init__(self) :
        self.library = []
        self.nHaps = 0
        self.split_states = []
        # self.randGen = jit_RandomBinary(1000) #Hard coding for now. 1000 seemed reasonable.

    def append(self, hap):
        self.library.append(hap)
        self.nHaps = len(self.library)

    def removeMissingValues(self):
        for hap in self.library:
            removeMissingValues(hap)
        # self.library = np.array(self.library)

    def asMatrix(self):
        return np.array(self.library)


    # def setup_split_states(self):
    #     nHaps, nLoci = self.library.shape
    #     for i in range(nLoci):
    #         zero_haps = []
    #         one_haps = []

    #         for hap in range(nHaps):
    #             if self.library[hap, i] == 0:
    #                 zero_haps.append(hap)
    #             else:
    #                 one_haps.append(hap)
    #         self.split_states.append((zero_haps, one_haps))


@njit
def removeMissingValues(hap):
    for i in range(len(hap)) :
        if hap[i] == 9:
            hap[i] = random.randint(0, 1)






