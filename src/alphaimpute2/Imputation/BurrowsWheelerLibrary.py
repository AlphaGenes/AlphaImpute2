
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
        self.library = []
        self.nHaps = 0

        self.individual_dictionary = dict()

        self.library_created = False

    def append(self, hap, ind = None):
        if self.library_created :
            print("WARNING: BW library has alread been created, new haplotypes cannot be added.")
        else:
            self.library.append(hap)
            self.nHaps = len(self.library)

            if ind is not None:
                idn = ind.idn
                current_haps = []
                if idn in self.individual_dictionary:
                    tmp_ind, current_haps = self.individual_dictionary[idn]
                current_haps += [len(self.library) -1]
                self.individual_dictionary[idn] = (ind, current_haps)

    def removeMissingValues(self):
        for hap in self.library:
            removeMissingValues(hap)


    def setup_library(self):
        self.removeMissingValues()
        self.library = createBWLibrary(np.array(self.library))
        self.library_created = True

        self.set_exclusions()

    def set_exclusions(self):
        if len(self.individual_dictionary) > 0:
            self.library.set_reverse_library()

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

@jitclass(jit_BurrowsWheelerLibrary_spec)
class jit_BurrowsWheelerLibrary():
    def __init__(self, a, d, nZeros, zeroOccNext, haps):
        self.a = a
        self.d = d
        self.nZeros = nZeros
        self.zeroOccNext = zeroOccNext
        self.haps = haps
        self.nHaps, self.nLoci = haps.shape

        self.reverse_library = np.full((1, 1), 0, dtype = np.int64)

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

        if lowerR >= upperR:
            return (-1, -1)
        else:
            return (lowerR, upperR)


    def get_null_state(self, index):
        return ((0, self.nZeros[index]), (self.nZeros[index], self.nHaps))


@njit
def createBWLibrary_with_d(haps):
    
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
    dZeros = np.full(nHaps, 0, dtype = np.int64)
    dOnes = np.full(nHaps, 0, dtype = np.int64)
    
    nZeros = 0
    nOnes = 0
    for j in range(nHaps):
        if haps[j, 0] == 0:
            zeros[nZeros] = j
            if nZeros == 0:
                dZeros[nZeros] = 0
            else:
                dZeros[nZeros] = 1
            nZeros += 1
        else:
            ones[nOnes] = j    
            if nOnes == 0:
                dOnes[nOnes] = 0
            else:
                dOnes[nOnes] = 1
            nOnes += 1
    if nZeros > 0:
        a[0:nZeros, 0] = zeros[0:nZeros]
        d[0:nZeros, 0] = dZeros[0:nZeros]

    if nOnes > 0:
        a[nZeros:nHaps, 0] = ones[0:nOnes]
        d[nZeros:nHaps, 0] = dOnes[0:nOnes]

    nZerosArray[0] = nZeros

    for i in range(1, nLoci) :
        zeros = np.full(nHaps, 0, dtype = np.int64)
        ones = np.full(nHaps, 0, dtype = np.int64)
        dZeros = np.full(nHaps, 0, dtype = np.int64)
        dOnes = np.full(nHaps, 0, dtype = np.int64)
    
        nZeros = 0
        nOnes = 0

        dZerosTmp = -1 #This is a hack.
        dOnesTmp = -1

        for j in range(nHaps) :

            dZerosTmp = min(dZerosTmp, d[j,i-1])
            dOnesTmp = min(dOnesTmp, d[j,i-1])
            if haps[a[j, i-1], i] == 0:
                zeros[nZeros] = a[j, i-1]
                dZeros[nZeros] = dZerosTmp + 1
                nZeros += 1
                dZerosTmp = nLoci
            else:
                ones[nOnes] = a[j, i-1]
                dOnes[nOnes] = dOnesTmp + 1
                nOnes += 1
                dOnesTmp = nLoci


        if nZeros > 0:
            a[0:nZeros, i] = zeros[0:nZeros]
            d[0:nZeros, i] = dZeros[0:nZeros]

        if nOnes > 0:
            a[nZeros:nHaps, i] = ones[0:nOnes]
            d[nZeros:nHaps, i] = dOnes[0:nOnes]
        nZerosArray[i] = nZeros


    #I'm going to be a wee bit sloppy in creating zeroOccNext
    #Not defined at 0 so start at 1.
    zeroOccNext = np.full(haps.shape, 0, dtype = np.int64)

    for i in range(0, nLoci-1):
        count = 0
        for j in range(0, nHaps):
            if haps[a[j, i], i+1] == 0:
                count += 1
            zeroOccNext[j, i] = count


    library = jit_BurrowsWheelerLibrary(a, d, nZerosArray, zeroOccNext, haps)
    return library

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


    library = jit_BurrowsWheelerLibrary(a, d, nZerosArray, zeroOccNext, haps)
    return library

# @jit(nopython=True, nogil=True)
# def getConsistentHaplotypes(bwLibrary, hap, start, stop):
#     a, d, nZeros, zeroOccNext, haps = bwLibrary.getValues()
#     nHaps = a.shape[0]
#     nLoci = a.shape[1]


#     intervals = np.full((nHaps, 2), 0, dtype = np.int64)
#     intervals_new = np.full((nHaps, 2), 0, dtype = np.int64)
    
#     nIntervals = 0
#     nIntervals_new = 0
    
#     #Haps go from 0 to nHaps-1. Loci go from start to stop-1 (inclusive).
#     #The first hap with one is nZeros. The last hap with zero is nZeros -1.
#     #Last loci is stop -1
#     #These are split out because they represent *distinct* haplotypes.
#     #Maybe could do this with tuple and list append but *shrug*.

#     if hap[stop-1] == 0 or hap[stop-1] == 9:
#         lowerR = 0
#         upperR = nZeros[stop-1]
#         if upperR >= lowerR:
#             intervals[nIntervals, 0] = lowerR
#             intervals[nIntervals, 1] = upperR
#             nIntervals += 1
    
#     if hap[stop-1] == 1 or hap[stop-1] == 9:
#         lowerR = nZeros[stop-1]
#         upperR = nHaps
#         if upperR >= lowerR:
#             intervals[nIntervals, 0] = lowerR
#             intervals[nIntervals, 1] = upperR
#             nIntervals += 1

#     #Python indexing is annoying.
#     #Exclude stop and stop-1, include start.
#     #Intervals are over haplotypes.
#     for i in range(stop-2, start-1, -1):
#         # print(intervals[0:nIntervals,:])

#         nIntervals_new = 0

#         #Doing it on interval seems to make marginally more sense.
#         for interval in range(nIntervals):

#             int_start = intervals[interval, 0]
#             int_end = intervals[interval, 1]

#             if hap[i] == 0 or hap[i] == 9:
#                 if int_start == 0:
#                     lowerR = 0
#                 else:
#                     lowerR = zeroOccNext[int_start-1, i+1] 
#                 upperR = zeroOccNext[int_end-1, i+1] #Number of zeros in the region. 
#                 if upperR > lowerR: #Needs to be greater than. Regions no longer inclusive.
#                     # print("Added 0:", int_start, int_end, "->>", lowerR, upperR)
#                     intervals_new[nIntervals_new, 0] = lowerR
#                     intervals_new[nIntervals_new, 1] = upperR
#                     nIntervals_new += 1
#             if hap[i] == 1 or hap[i] == 9:

#                 # of ones between 0 and k (k inclusive) is k+1 - number of zeros.

#                 if int_start == 0:
#                     lowerR = nZeros[i]
#                 else:
#                     lowerR = nZeros[i] + (int_start - zeroOccNext[int_start-1, i+1]) 
#                 upperR = nZeros[i] + (int_end - zeroOccNext[int_end-1, i+1])
#                 if upperR > lowerR:
#                     # print("Added 1:", int_start, int_end, "->>", lowerR, upperR)
#                     intervals_new[nIntervals_new, 0] = lowerR
#                     intervals_new[nIntervals_new, 1] = upperR
#                     nIntervals_new += 1
#                 # else:
#                     # print(i, "interval rejected:", int_start, int_end, "->", upperR, lowerR)
#         #This is basically intervals = intervals_new
#         for j in range(nIntervals_new):
#             intervals[j, 0] = intervals_new[j, 0]
#             intervals[j, 1] = intervals_new[j, 1]
#         nIntervals = nIntervals_new
#         # print("Finished", i, "->", nIntervals)
#     # print(intervals[0:nIntervals,:])

#     hapIndexes = np.full((nHaps, 2), 0, dtype = np.int64)
#     nHapsAssigned = 0
#     for i in range(nIntervals):

#         int_start = intervals[i, 0]
#         int_end = intervals[i, 1]


#         hapIndexes[nHapsAssigned, 0] = a[int_start,start]
#         hapIndexes[nHapsAssigned, 1] = int_end - int_start
#         nHapsAssigned +=1

#     return (nHapsAssigned, hapIndexes)

# def printSortAt(loci, library):
#     a, d, nZeros, zeroOccNext, haps = library.getValues()
#     vals = haps[a[:,loci],:]
#     for i in range(vals.shape[0]):
#         print(i, ' '.join(map(str, vals[i,:])) )
    
#     # print(zeroOccNext[:,:])

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

# # printSortAt(0, bwlib.library)
# printSortAt(6, bwlib.library); print("")
# printSortAt(5, bwlib.library); print("")
# printSortAt(4, bwlib.library); print("")
# printSortAt(3, bwlib.library); print("")
# printSortAt(2, bwlib.library); print("")
# printSortAt(1, bwlib.library); print("")
# printSortAt(0, bwlib.library); print("")


# print(bwlib.library.get_null_state(0))
# print(bwlib.library.update_state((0, 13), 1))

# bwlib.library.set_reverse_library()
# print(bwlib.library.a)

# print(bwlib.library.reverse_library)


# # print(bwlib.getHaplotypeMatches(haplotype = np.array([0, 0, 0], dtype = np.int8), start = 0, stop = 3))
# # tmp = (bwlib.getHaplotypeMatches(haplotype = np.array([9, 9, 9, 9, 9, 9, 9], dtype = np.int8), start = 0, stop = 7))
# # for key, value in tmp:
# #     print(key, value)
