import numpy as np
from numba import jit, float32

import concurrent.futures
from itertools import repeat

import datetime

from . import Imputation
from . import FamilyParentPhasing

from ..tinyhouse import ProbMath


# np.core.arrayprint._line_width = 200
# np.set_printoptions(precision=4, suppress=True, edgeitems=3)

# Boiler plate profiler code to make this play nicely with Kernprofiler
try:
    profile
except:
    def profile(x): 
        return x

# Decorator to add in timings with custom text.
# For more reading see, maybe https://stackoverflow.com/questions/5929107/decorators-with-parameters
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


@time_func("Recomb Estimation")
@profile
def pedigreeRecombEstimate(pedigree, args, cutoff):
    # This function peels down a pedigree; i.e. it finds which regions an individual inherited from their parents, and then fills in the individual's anterior term using that information.
    # To do this peeling, individual's genotypes should be set to poster+penetrance; parent's genotypes should be set to All.
    # Since parents may be shared across families, we set the parents seperately from the rest of the family. 
    # We then set the child genotypes, calculate the segregation estimates, and calculate the anterior term on a family by family basis (set_segregation_peel_down).

    for generation in pedigree.generations:
        for parent in generation.parents:
            parent.peeling_view.setGenotypesAll(cutoff)

        if args.maxthreads <= 1:
            for family in generation.families:
                set_recombination_estimates_family(family, cutoff, repeat(args.length))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.maxthreads) as executor:
                executor.map(set_recombination_estimates_family, generation.families, repeat(cutoff), repeat(args.length))



def set_recombination_estimates_family(family, cutoff, map_length):
    sire = family.sire.peeling_view
    dam = family.dam.peeling_view
    offspring = [ind.peeling_view for ind in family.offspring]
    set_segregation_peel_down_jit(sire, dam, offspring, cutoff, map_length)

@jit(nopython=True, nogil = True)
def set_recombination_estimates_family_jit(sire, dam, offspring, cutoff, map_length):
    nLoci = len(sire.genotypes)
    transmissionRate = jit_recombScore(map_length/nLoci)

    nOffspring = len(offspring)
    for child in offspring:
        # We don't need to re-set calculate the posterior here, since the posterior value is constant for the peel down pass.
        child.setGenotypesPosterior(cutoff)

    nOffspring = len(offspring)
    for child in offspring:
        estimateLocusRecombination(child, sire, dam)



spec = OrderedDict()
spec['score'] = float32[:,:]
spec['score_mat'] = float32[:,:]
spec['score_pat'] = float32[:,:]
spec['mat'] = float32[:,:,:]

@jitclass(spec)
class jit_recombScore(object):
    def __init__(self, transmissionRate):
        self.score = np.array([[0, 1, 1, 2],
                      [1, 0, 2, 1],
                      [1, 2, 0, 1],
                      [2, 1, 1, 0]], dtype = np.float32)
    
    
        self.score_pat = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [1, 1, 0, 0],
                          [1, 1, 0, 0]], dtype = np.float32)

        self.score_mat = np.array([[0, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 0, 1, 0]], dtype = np.float32)
        self.mat = np.full((len(transmissionRate), 4, 4), 0, dtype = np.float32)
        for i in range(len(transmissionRate)):
            e = transmissionRate[i]
            self.mat[i,:,:] = np.array([[(1-e)**2,  (1-e)*e,    (1-e)*e,    e**2],
                                [(1-e)*e,   (1-e)**2,   e**2,       (1-e)*e],
                                [(1-e)*e,   e**2,       (1-e)**2,   (1-e)*e],
                                [e**2,      (1-e)*e,    (1-e)*e,    (1-e)**2]])



@jit(nopython=True, nogil = True)
def set_recombination_estimates(ind, sire, dam, recombScore):

    nLoci = len(ind.genotypes)

    pointEstimates = np.full((4, nLoci), 1, dtype = np.float32)
    fillPointEstimates(pointEstimates, ind, sire, dam)

    smoothedEstimates, forwardSeg, backwardSeg = smoothPointSeg(pointEstimates, 1.0/nLoci) # This is where different map lengths could be added.

    for i in range(nLoci -1):
        ind.recomb[i], ind.recomb_mat[i], ind.recomb_pat[i] = estimateLocusRecombination(pointSeg, forwardSeg, backwardSeg, recombScore, i)




@jit(nopython=True, nogil=True)
def estimateLocusRecombination(pointSeg, forwardSeg, backwardSeg, recombScore, locus):
    # Estimates the transmission rate between locus and locus + 1.
    val_current = np.full(4, 0, dtype = np.float32)
    val_next = np.full(4, 0, dtype = np.float32)

    for i in range(4):
        val_current[i] = pointSeg[i,locus]*forwardSeg[i,locus]

    for i in range(4):
        val_next[i] = pointSeg[i,locus+1]*backwardSeg[i,locus+1]

    norm_1D(val_current)
    norm_1D(val_next)
    
    # Now create joint probabilities.

    mat = np.full((4, 4), 0, dtype = np.float32)
    for i in range(4):
        for j in range(4):
            mat[i,j] = recombScore.mat[locus,i,j]


    # Now create joint probabilities.
    for i in range(4):
        for j in range(4):
            mat[i,j] *= val_current[i]*val_next[j]

    norm_2D(mat)
    error = 0
    error_mat = 0
    error_pat = 0
    for i in range(4):
        for j in range(4):
            error += mat[i,j]*recombScore.score[i,j]
            error_mat += mat[i,j]*recombScore.score_pat[i,j]
            error_pat += mat[i,j]*recombScore.score_mat[i,j]

    return(error, error_mat, error_pat)


@jit(nopython=True, nogil = True)
def fillPointEstimates(pointEstimates, ind, sire, dam):
    nLoci = pointEstimates.shape[1]
    e = 0.01 # Assume 1% genotyping error.
    for i in range(nLoci):
        # Let's do sire side.
        # I'm going to assume we've already peeled down.
        
        sirehap0 = sire.haplotypes[0][i]
        sirehap1 = sire.haplotypes[1][i]
        damhap0 = dam.haplotypes[0][i]
        damhap1 = dam.haplotypes[1][i]

        # There's an extra edge case where both the child is heterozygous, unphased, and both the parent's haplotypes are phased.
        if ind.genotypes[i] == 1 and ind.haplotypes[0][i] == 9 and ind.haplotypes[0][i] == 9 :
            if sirehap0 != 9 and sirehap1 != 9 and damhap0 != 9 and damhap1 != 9:
                # This is ugly, but don't have a better solution.

                if sirehap0 + damhap0 == 1:
                    pointEstimates[0,i] *= 1-e
                else:
                    pointEstimates[0,i] *= e

                if sirehap0 + damhap1 == 1:
                    pointEstimates[1,i] *= 1-e
                else:
                    pointEstimates[1,i] *= e

                if sirehap1 + damhap0 == 1:
                    pointEstimates[2,i] *= 1-e
                else:
                    pointEstimates[2,i] *= e

                if sirehap1 + damhap1 == 1:
                    pointEstimates[3,i] *= 1-e
                else:
                    pointEstimates[3,i] *= e

        if ind.haplotypes[0][i] != 9:
            indhap = ind.haplotypes[0][i]

            # If both parental haplotypes are non-missing, and
            # not equal to each other, then this is an informative marker.
            if sirehap0 != 9 and sirehap1 != 9 and sirehap0 != sirehap1:

                if indhap == sirehap0 :
                    pointEstimates[0,i] *= 1-e
                    pointEstimates[1,i] *= 1-e
                    pointEstimates[2,i] *= e
                    pointEstimates[3,i] *= e

                if indhap == sirehap1:
                    pointEstimates[0,i] *= e
                    pointEstimates[1,i] *= e
                    pointEstimates[2,i] *= 1-e
                    pointEstimates[3,i] *= 1-e

        if ind.haplotypes[1][i] != 9:
            indhap = ind.haplotypes[1][i]
    
            if damhap0 != 9 and damhap1 != 9 and damhap0 != damhap1:

                if indhap == damhap0:
                    pointEstimates[0,i] *= 1-e
                    pointEstimates[1,i] *= e
                    pointEstimates[2,i] *= 1-e
                    pointEstimates[3,i] *= e
                
                if indhap == damhap1 :
                    pointEstimates[0,i] *= e
                    pointEstimates[1,i] *= 1-e
                    pointEstimates[2,i] *= e
                    pointEstimates[3,i] *= 1-e

@jit(nopython=True, nogil=True, locals={'e': float32, 'e2':float32, 'e1e':float32, 'e2i':float32})
def smoothPointSeg(pointSeg, transmission):

    nLoci = pointSeg.shape[1] 

    # Seg is the output, and is a copy of pointseg.
    seg = np.full(pointSeg.shape, .25, dtype = np.float32)
    forwardSeg = np.full(pointSeg.shape, .25, dtype = np.float32)
    backwardSeg = np.full(pointSeg.shape, .25, dtype = np.float32)

    for i in range(nLoci):
        for j in range(4):
            seg[j,i] = pointSeg[j,i]

    # Variables.
    tmp = np.full(4, 0, dtype = np.float32)
    new = np.full(4, 0, dtype = np.float32)
    prev = np.full(4, .25, dtype = np.float32)

    # Transmission constants.
    e = transmission
    e2 = e**2
    e1e = e*(1-e)
    e2i = (1.0-e)**2

    for i in range(1, nLoci):
        # Combine previous estimate with previous pointseg and then transmit forward.
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i-1]
        
        norm_1D(tmp)

        # Father/Mother; Paternal/Maternal
        # !                  fm  fm  fm  fm 
        # !segregationOrder: pp, pm, mp, mm

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[j,i] *= new[j]
            forwardSeg[j,i] = new[j]

        prev = new

    prev = np.full((4), .25, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        
        for j in range(4):
            tmp[j] = prev[j]*pointSeg[j,i+1]
        
        norm_1D(tmp)

        new[0] = e2*tmp[3] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[0] 
        new[1] = e2*tmp[2] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[1] 
        new[2] = e2*tmp[1] + e1e*(tmp[0] + tmp[3]) + e2i*tmp[2] 
        new[3] = e2*tmp[0] + e1e*(tmp[1] + tmp[2]) + e2i*tmp[3] 

        for j in range(4):
            seg[j,i] *= new[j]
            backwardSeg[j,i] = new[j]

        prev = new
    
    for i in range(nLoci):
        norm_1D(seg[:,i])

    return seg, forwardSeg, backwardSeg


############
#
#  Numba matrix functions
#
###########

@jit(nopython=True, nogil = True)
def norm_1D(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total

@jit(nopython=True, nogil=True)
def norm_2D(values) :
    count = 0
    for i in range(4):
        for j in range(4):
            count += values[i, j]
    for i in range(4):
        for j in range(4):
            values[i,j] /= count

