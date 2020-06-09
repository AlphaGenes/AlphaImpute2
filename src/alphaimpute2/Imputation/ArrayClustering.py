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
        self.n_observed_genotypes = 0
        self.sum_genotype = None
        self.individuals = []

    def distance(self, individual):
        dist = snp_array_distance(self.genotypes, individual.genotypes !=9)
        return dist

    def add(self, individual):
        if not self.fixed:
            if self.sum_genotype is None:
                self.sum_genotype = (individual.genotypes!= 9).astype(int)
                self.n_observed_genotypes = 1
            else:
                self.sum_genotype += (individual.genotypes != 9)
                self.n_observed_genotypes += 1

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
            self.n_observed_genotypes = 0

            for individual in individuals:
                self.add(individual)
    @property
    def genotypes(self):
        return np.floor(self.sum_genotype/self.n_observed_genotypes + 0.1)
    

    @property
    def n_markers(self):
        return int(np.sum(self.genotypes))
    
    @property
    def n_ind(self):
        return len(self.individuals)

    def copy(self):
        new_array = SNP_Array()
        new_array.fixed = self.fixed
        new_array.n_observed_genotypes = self.n_observed_genotypes
        new_array.sum_genotype = self.sum_genotype.copy()
        return new_array

class ArrayContainer():
    def __init__(self):
        self.arrays = []

    def __iter__(self):
        sorted_arrays = sorted(self.arrays, key = lambda array: array.n_markers, reverse = True)
        for array in sorted_arrays:
            yield array


    def __len__(self):
        return len(self.arrays)


    def append(self, array):
        self.arrays.append(array)


    def filter(self, min_individuals = None, min_markers = None):
        if min_individuals is not None:
            self.arrays = [array for array in self.arrays if array.n_ind > min_individuals]

        if min_markers is not None:
            self.arrays = [array for array in self.arrays if array.n_markers > min_markers]

    def fix_and_clear(self):
        for array in self.arrays:
            array.fix_and_clear()


    def reduce_arrays(self, n_arrays):

        removed_individuals = []

        if len(self.arrays) > n_arrays:
            array_list = sorted(self.arrays, key = lambda array: array.n_ind, reverse = True)
            self.arrays = array_list[0:n_arrays]
            
            for array in array_list[n_arrays:]:
                removed_individuals += array.individuals
            return removed_individuals

        return removed_individuals

    def remove_duplicate_arrays(self, distance_threshold = 0.99):
        # Remove arrays that are highly similar.
        arrays_to_remove = []
        for array_1 in self.arrays:
            for array_2 in self.arrays:
                if array_1 not in arrays_to_remove and array_2 not in arrays_to_remove:
                    if array_1 is not array_2:
                        distance = snp_array_distance(array_1.genotypes, array_2.genotypes)
                        if distance > distance_threshold:
                            arrays_to_remove.append(array_2)
        
        for array in arrays_to_remove:
            self.arrays.remove(array) 



    def copy(self):
        new_container = ArrayContainer()
        new_container.arrays = [array.copy() for array in self.arrays]
        new_container.fix_and_clear()
        return new_container


    def write_out_arrays(self, file_name):

        with open(file_name, "w+") as f:
            for i, array in enumerate(self):
                for individual in array.individuals :
                    f.write(f"{individual.idx} {i}\n")

    def __str__(self):
        output = ""
        output += "Array \t N_Ind \t N_Markers"
        for i, array in enumerate(self):
            n_markers = array.n_markers
            output += f"\n{i+1}\t {len(array.individuals)}\t {n_markers}"
        return output

def cluster_individuals_by_array(individuals, min_frequency) :

    arrays = ArrayContainer()
    assign_individuals_to_array(individuals, arrays, allow_new_arrays = True)

    cutoff = min_frequency

    arrays.filter(min_individuals = cutoff)
    arrays.fix_and_clear()

    assign_individuals_to_array(individuals, arrays, allow_new_arrays = False)
    return arrays


def assign_individuals_to_array(individuals, arrays, allow_new_arrays = False, max_arrays = 20):

    skipped_individuals = []

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
            if len(arrays) > 2*max_arrays:
                removed_individuals = arrays.reduce_arrays(max_arrays)
                skipped_individuals += removed_individuals

            new_centroid = SNP_Array()
            new_centroid.add(individual)
            arrays.append(new_centroid)            

    if len(skipped_individuals) > 0:
        assign_individuals_to_array(skipped_individuals, arrays, allow_new_arrays = False)

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

    # Clean all arrays and set individuals to empty
    for array in new_arrays:
        arrays.append(array)

    arrays.fix_and_clear()
    arrays.remove_duplicate_arrays(distance_threshold = 0.99)
    assign_individuals_to_array(individuals, arrays, allow_new_arrays = False)

    # Remove empty arrays
    arrays.filter(min_individuals = 0)

    return arrays

def create_array_subset(individuals, original_arrays, min_markers = 0, min_individuals = 0):
    new_arrays = original_arrays.copy()
    assign_individuals_to_array(individuals, new_arrays, allow_new_arrays = False)
    new_arrays.filter(min_individuals = min_individuals, min_markers = min_markers)
    return new_arrays

