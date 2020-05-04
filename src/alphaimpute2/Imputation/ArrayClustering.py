from numba import njit
import numpy as np


@njit
def snp_array_distance(g1, g2):
    """Fraction of loci that agree"""
    eps = 0.5 # rounds 0.5 or above to 1
    return np.mean((np.floor(g1+eps) == np.floor(g2+eps)))



class SNP_Array(object):

    def __init__(self, individual):
        self.sum_genotype = (individual.genotypes!= 9).astype(int)
        self.n = 1

        self.individuals = [individual]
        self.fixed = False

    def distance(self, individual):
        dist = snp_array_distance(self.sum_genotype/self.n, individual.genotypes !=9)
        return dist

    def add(self, individual):
        if not self.fixed:
            self.sum_genotype += (individual.genotypes != 9)
            self.n += 1

        self.individuals.append(individual)

    def fix_and_clear(self):
        self.fixed = True
        self.individuals = []


def cluster_individuals_by_array(individuals, min_frequency) :

    arrays = []
    assign_individuals_to_array(individuals, arrays, allow_new_arrays = True)

    cutoff = len(individuals)*min_frequency

    arrays = [array for array in arrays if array.n > cutoff]

    for array in arrays:
        array.fix_and_clear()

    assign_individuals_to_array(individuals, arrays, allow_new_arrays = False)

    return arrays


def assign_individuals_to_array(individuals, arrays, allow_new_arrays = False):

    if allow_new_arrays:
        threshold = 0.9
    else:
        # As long as arrays is not empty, this will allways assign individuals to an array.
        threshold = 0

    for individual in individuals:
        maxd = 0.0
        max_array = None
        for i, array in enumerate(arrays):
            d = array.distance(individual)
            if d > maxd and d > threshold:
                maxd = d
                max_array = array
        
        if max_array is not None:
            max_array.add(individual)
        else:
            new_centroid = SNP_Array(individual)
            arrays.append(new_centroid)            
    return arrays



