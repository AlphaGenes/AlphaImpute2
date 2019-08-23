
import random
import numpy as np
import numba
from numba import njit, jit, int8, int32,int64, boolean, deferred_type, optional, jitclass, float32
from collections import OrderedDict

try:
    profile
except:
    def profile(x): 
        return x

#####################################
###                               ###
###       Burrows Wheeler         ###
###                               ###
#####################################

class BurrowsWheelerLibrary():
    def __init__(self):
        self.haplotypes = []
        self.library = None
        self.reverse_library = None
        self.nHaps = 0
        self.bw_loci = None
        self.individual_dictionary = dict()
        self.library_created = False

    def append(self, hap, ind = None):
        if self.library_created :
            print("WARNING: BW library has alread been created, new haplotypes cannot be added.")
        else:
            self.haplotypes.append(hap)
            self.nHaps = len(self.haplotypes)

            if ind is not None:
                idn = ind.idn
                current_haps = []
                if idn in self.individual_dictionary:
                    tmp_ind, current_haps = self.individual_dictionary[idn]
                current_haps += [len(self.haplotypes) -1]
                self.individual_dictionary[idn] = (ind, current_haps)

    def removeMissingValues(self):
        for hap in self.haplotypes:
            removeMissingValues(hap)


    def setup_library(self, loci = None):
        self.removeMissingValues()
        hap_array = np.array(self.haplotypes)
        if loci is not None:
            bw_loci = loci
        else:
            bw_loci = list(range(hap_array.shape[1]))
        
        bw_loci = np.array(bw_loci, dtype = np.int64)
        self.library = jit_BurrowsWheelerLibrary(hap_array[:,bw_loci], hap_array, bw_loci)

        self.library_created = True

        self.set_exclusions()

    def set_exclusions(self):
        if len(self.individual_dictionary) > 0:
            for ind, haps in self.individual_dictionary.values():
                exclusion = self.library.reverse_library[haps,:]
                ind.phasing_view.set_own_haplotypes(exclusion)


@njit
def removeMissingValues(hap):
    for i in range(len(hap)) :
        if hap[i] == 9:
            if random.random() > .5:
                hap[i] = 1
            else:
                hap[i] = 0


jit_BurrowsWheelerLibrary_spec = OrderedDict()
jit_BurrowsWheelerLibrary_spec['a'] = int64[:,:]
jit_BurrowsWheelerLibrary_spec['d'] = int64[:,:]
jit_BurrowsWheelerLibrary_spec['reverse_library'] = int64[:,:]
jit_BurrowsWheelerLibrary_spec['zeroOccNext'] = int64[:,:]
jit_BurrowsWheelerLibrary_spec['nZeros'] = int64[:]
jit_BurrowsWheelerLibrary_spec['haps'] = int8[:,:]
jit_BurrowsWheelerLibrary_spec['nHaps'] = int64
jit_BurrowsWheelerLibrary_spec['nLoci'] = int64

jit_BurrowsWheelerLibrary_spec['full_haps'] = int8[:,:]
jit_BurrowsWheelerLibrary_spec['full_nLoci'] = int64

jit_BurrowsWheelerLibrary_spec['loci'] = int64[:]

@jitclass(jit_BurrowsWheelerLibrary_spec)
class jit_BurrowsWheelerLibrary():
    def __init__(self, haps, full_haps, loci):
        self.full_haps = full_haps
        self.loci = loci
        self.full_nLoci = full_haps.shape[1]
        self.a, self.d, self.nZeros, self.zeroOccNext, self.haps = createBWLibrary(haps)
        self.nHaps, self.nLoci = self.haps.shape

        self.reverse_library = np.full((1, 1), 0, dtype = np.int64)
        self.set_reverse_library()

    def get_true_index(self, index):
        return self.loci[index]

    def getValues(self):
        return (self.a, self.d, self.nZeros, self.zeroOccNext, self.haps)

    def set_reverse_library(self):

        self.reverse_library = np.full(self.a.shape, 0, dtype = np.int64)

        for i in range(self.nLoci):
            for j in range(self.nHaps):

                self.reverse_library[self.a[j, i], i] = j


    def update_state(self, state, index):
        
        # Note: index needs to be greater than 1.

        int_start, int_end = state
        if int_end - int_start <= 0:
            return ((-1, -1), (-1, -1))

        # zero_next_start = self.zeroOccNext[int_start-1, index-1]
        # zero_next_end = self.zeroOccNext[int_end-1, index-1]

        # zeros_index = self.nZeros[index]

        # Set up val_0

        if int_start == 0:
            lowerR = 0
        else:
            lowerR = self.zeroOccNext[int_start-1, index-1]
        upperR = self.zeroOccNext[int_end-1, index-1] #Number of zeros in the region. 
    
        if lowerR >= upperR:
            vals_0 = (-1, -1)
        else:
            vals_0 = (lowerR, upperR)

        # set up val_1

        if int_start == 0:
            lowerR = self.nZeros[index]
        else:
            lowerR = self.nZeros[index] + (int_start - self.zeroOccNext[int_start-1, index-1]) 
        upperR = self.nZeros[index] + (int_end - self.zeroOccNext[int_end-1, index-1])

        if lowerR >= upperR:
            vals_1 = (-1, -1)
        else:
            vals_1 = (lowerR, upperR)

        return (vals_0, vals_1)

    def update_state_value(self, state, index, value):
        
        # Note: index needs to be greater than 1.

        int_start, int_end = state
        if int_end - int_start <= 0:
            return (-1, -1)

        if value == 0:
            if int_start == 0:
                lowerR = 0
            else:
                lowerR = self.zeroOccNext[int_start-1, index-1] 
            upperR = self.zeroOccNext[int_end-1, index-1] #Number of zeros in the region. 

        if value == 1:

            # of ones between 0 and k (k inclusive) is k+1 - number of zeros.

            if int_start == 0:
                lowerR =self.nZeros[index]
            else:
                lowerR =self.nZeros[index] + (int_start - self.zeroOccNext[int_start-1, index-1]) 
            upperR =self.nZeros[index] + (int_end - self.zeroOccNext[int_end-1, index-1])

        if lowerR < upperR:
            return (lowerR, upperR)

        return (-1, -1)


    def get_null_state(self, index):
        return ((0, self.nZeros[index]), (self.nZeros[index], self.nHaps))


@njit
def createBWLibrary(haps):
    
    #Definitions.
    # haps : a list of haplotypes
    # a : an ordering of haps in lexographic order.
    # d : the length of the longest match between each (sorted) haplotype and the previous (sorted) haplotype.

    nHaps = haps.shape[0]
    nLoci = haps.shape[1]
    a = np.full(haps.shape, 0, dtype = np.int64)
    d = np.full(haps.shape, 0, dtype = np.int64)

    nZerosArray = np.full(nLoci, 0, dtype = np.int64)

    zeros = np.full(nHaps, 0, dtype = np.int64)
    ones = np.full(nHaps, 0, dtype = np.int64)
    
    nZeros = 0
    nOnes = 0
    for j in range(nHaps):
        if haps[j, 0] == 0:
            zeros[nZeros] = j
            nZeros += 1
        else:
            ones[nOnes] = j    
            nOnes += 1
    if nZeros > 0:
        a[0:nZeros, 0] = zeros[0:nZeros]

    if nOnes > 0:
        a[nZeros:nHaps, 0] = ones[0:nOnes]

    nZerosArray[0] = nZeros
    zeroOccNext = np.full(haps.shape, 0, dtype = np.int64)

    for i in range(1, nLoci) :
        
        # zeros[:] = 0 # We should be safe without doing this; i.e. we only use the part of zeros that we set.
        # ones[:] = 0  # We should be safe without doing this; i.e. we only use the part of zeros that we set.    
        nZeros = 0
        nOnes = 0

        z_occ_count = 0
        for j in range(nHaps) :

            if haps[a[j, i-1], i] == 0:
                zeros[nZeros] = a[j, i-1]
                nZeros += 1
                z_occ_count += 1
            else:
                ones[nOnes] = a[j, i-1]
                nOnes += 1

            # Updating zeroOccNext in a single pass.
            zeroOccNext[j, i-1] = z_occ_count

        if nZeros > 0:
            a[0:nZeros, i] = zeros[0:nZeros]

        if nOnes > 0:
            a[nZeros:nHaps, i] = ones[0:nOnes]
        nZerosArray[i] = nZeros
        

    # for i in range(0, nLoci-1):
    #     count = 0
    #     for j in range(0, nHaps):
    #         if haps[a[j, i], i+1] == 0:
    #             count += 1
    #         zeroOccNext[j, i] = count


    return a, d, nZerosArray, zeroOccNext, haps


example_bw_library = None
def get_example_library():
    global example_bw_library
    if example_bw_library is None:
        hapLib = [np.array([1, 0, 0, 0, 0, 0, 1], dtype = np.int8),  # 0 
                  np.array([0, 1, 0, 0, 0, 1, 0], dtype = np.int8),  # 1
                  np.array([0, 1, 0, 0, 0, 1, 0], dtype = np.int8),  # 2
                  np.array([0, 1, 0, 0, 0, 1, 0], dtype = np.int8),  # 3
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 4
                  np.array([1, 1, 1, 0, 0, 0, 0], dtype = np.int8),  # 5
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 6
                  np.array([1, 1, 1, 0, 1, 0, 0], dtype = np.int8),  # 7
                  np.array([0, 0, 0, 1, 0, 0, 0], dtype = np.int8),  # 8
                  np.array([0, 1, 1, 1, 0, 0, 0], dtype = np.int8),  # 9
                  np.array([0, 1, 1, 1, 0, 0, 0], dtype = np.int8),  # 10
                  np.array([1, 1, 0, 1, 0, 0, 0], dtype = np.int8),  # 11
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 12
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 13
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 14
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8),  # 15
                  np.array([0, 0, 1, 0, 1, 0, 0], dtype = np.int8)]  # 16
        example_bw_library = BurrowsWheelerLibrary()
        for hap in hapLib:
            example_bw_library.append(hap)
        example_bw_library.setup_library()

    return example_bw_library

def run_example():
    bwlib = get_example_library()
    def printSortAt(loci, library):
        a, d, nZeros, zeroOccNext, haps = library.getValues()
        vals = haps[a[:,loci],:]
        for i in range(vals.shape[0]):
            print(i, ' '.join(map(str, vals[i,:])) )
        
        # print(zeroOccNext[:,:])


    # printSortAt(0, bwlib.library)
    printSortAt(6, bwlib.library); print("")
    printSortAt(5, bwlib.library); print("")
    printSortAt(4, bwlib.library); print("")
    printSortAt(3, bwlib.library); print("")
    printSortAt(2, bwlib.library); print("")
    printSortAt(1, bwlib.library); print("")
    printSortAt(0, bwlib.library); print("")


    print(bwlib.library.get_null_state(0))
    print(bwlib.library.update_state((0, 13), 1))

    bwlib.library.set_reverse_library()
    print(bwlib.library.a)

    print(bwlib.library.reverse_library)

# run_example()


