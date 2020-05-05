from numba import njit
import numpy as np


@njit
def snp_array_distance(g1, g2):

    # g1 should have 90% of the markers that g2 has.
    # g2 should have 90% of the markers that g1 has.
    eps = 0.1

    called_1 = np.floor(g1 + eps)
    called_2 = np.floor(g2 + eps)

    n_1 = np.sum(called_1 == 1)
    n_2 = np.sum(called_2 == 1)
    overlap = np.sum((called_1 == 1) & (called_2 == 1))

    # If either group is fully missing and the other is not, return 0

    if n_1 == 0 and n_2 != 0:
        return 0

    if n_1 != 0 and n_2 == 0:
        return 0

    if n_1 == 0 and n_2 == 0:
        return 1

    return min(overlap/n_1, overlap/n_2)


class SNP_Array(object):

    def __init__(self):
        self.fixed = False
        self.n = 0
        self.sum_genotype = None
        self.individuals = []

    def distance(self, individual):
        dist = snp_array_distance(self.sum_genotype/self.n, individual.genotypes !=9)
        return dist

    def add(self, individual):
        if not self.fixed:
            if self.sum_genotype is None:
                self.sum_genotype = (individual.genotypes!= 9).astype(int)
                self.n = 1
            else:
                self.sum_genotype += (individual.genotypes != 9)
                self.n += 1

        self.individuals.append(individual)

    def fix_and_clear(self):
        self.fixed = True
        self.individuals = []

    def refresh_information(self):
        individuals = self.individuals
        
        if len(individuals) > 0:
            self.fixed = False
            self.individuals = []
            self.sum_genotype = None
            self.n = 0

            for individual in individuals:
                self.add(individual)

    @property
    def n_markers(self):
        return int(np.sum(np.floor(self.sum_genotype/self.n + 0.1)))
    

def cluster_individuals_by_array(individuals, min_frequency) :

    arrays = []
    assign_individuals_to_array(individuals, arrays, allow_new_arrays = True)

    cutoff = min_frequency

    arrays = [array for array in arrays if array.n > cutoff]

    for array in arrays:
        array.fix_and_clear()

    assign_individuals_to_array(individuals, arrays, allow_new_arrays = False)

    return arrays


def assign_individuals_to_array(individuals, arrays, allow_new_arrays = False):

    if allow_new_arrays:
        threshold = 0.8
    else:
        # As long as arrays is not empty, this will allways assign individuals to an array.
        threshold = 0

    for i, individual in enumerate(individuals):
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
            new_centroid = SNP_Array()
            new_centroid.add(individual)
            arrays.append(new_centroid)            
    
    print_array_list(arrays)
    return arrays


def update_arrays(arrays):
    new_arrays = []
    
    # Collect all of the individuals
    individuals = []
    for array in arrays:
        individuals += array.individuals

    # Create new arrays to store updated (post-pedigree imputation) information
    for array in arrays:
        new_array = SNP_Array()
        for individual in array.individuals:
            new_array.add(individual)
        new_arrays.append(new_array)

    # Remove arrays that are highly similar.
    arrays_to_remove = []
    for array_1 in arrays:
        for array_2 in arrays:
            if array_1 not in arrays_to_remove and array_2 not in arrays_to_remove:
                if array_1 is not array_2:
                    distance = snp_array_distance(array_1.sum_genotype/array_1.n, array_2.sum_genotype/array_2.n)
                    if distance > 0.99:
                        arrays_to_remove.add(array_2)
    
    for array in arrays_to_remove:
        arrays.remove(array) 

    # Clean all arrays and set individuals to empty
    arrays += new_arrays
    for array in arrays:
        array.fix_and_clear()

    assign_individuals_to_array(individuals, arrays, allow_new_arrays = False)

    # Remove empty arrays
    arrays[:] = [array for array in arrays if len(array.individuals) > 0]

    return arrays


def print_array_list(arrays):
    print("")
    print("Array \t N_Ind \t N_Markers")

    sorted_arrays = sorted(arrays, key = lambda array: array.n_markers, reverse = True)

    for i, array in enumerate(sorted_arrays):
        n_markers = array.n_markers
        print(i, "\t", len(array.individuals), "\t", n_markers)
