from ..tinyhouse import Pedigree

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass


import numba
from numba import jit, int8, int64, boolean, optional, float32
from collections import OrderedDict
import numpy as np
import hashlib
from . import Imputation

if "profile" not in locals():

    def profile(x):
        return x


class AlphaImputeIndividual(Pedigree.Individual):
    def __init__(self, idx, idn):
        super().__init__(idx, idn)

        self.reverse_view = None
        self.backward_information = None

        self.marker_score = None
        self.population_imputation_target = True  # Default value.

        # There may be times that we want to store original genotype and haplotype information.
        self.original_genotypes = None
        self.original_haplotypes = None

        self.current_haplotypes = None

        self.masked_sire = None
        self.masked_dam = None

        seed = self.get_random_seed()
        # self.random_generator = np.random.RandomState(seed)
        self.random_generator = np.random.default_rng(
            seed
        )  # Moving over to the newer generator.
        self.map_length = None

    def copy(self):
        new_ind = AlphaImputeIndividual(self.idx + "_copy", -1)
        new_ind.genotypes = self.genotypes.copy()
        new_ind.haplotypes = (self.haplotypes[0].copy(), self.haplotypes[1].copy())

        return new_ind

    def get_random_seed(self):
        as_encoded_string = str(self.idx).encode("utf-8")
        hash_val = hashlib.sha224(as_encoded_string).digest()
        return int.from_bytes(hash_val, "little") % (2**32)

    def mask_parents(self):
        self.masked_sire = self.sire
        self.masked_dam = self.dam

        if self.sire is not None:
            self.sire.offspring.remove(self)

        if self.dam is not None:
            self.dam.offspring.remove(self)

        self.sire = None
        self.dam = None

    def unmask_parents(self):
        self.sire = self.masked_sire
        self.dam = self.masked_dam

        if self.sire is not None:
            self.sire.offspring.append(self)

        if self.dam is not None:
            self.dam.offspring.append(self)

    def set_original_genotypes(self):
        self.original_genotypes = self.genotypes.copy()
        self.original_haplotypes = [
            self.haplotypes[0].copy(),
            self.haplotypes[1].copy(),
        ]

    def restore_original_genotypes(self):
        self.genotypes[:] = self.original_genotypes
        self.haplotypes[0][:] = self.original_haplotypes[0]
        self.haplotypes[1][:] = self.original_haplotypes[1]

        self.original_genotypes = None
        self.original_haplotypes = None

    @property
    def percent_phased(self):
        return np.mean((self.haplotypes[0] != 9) & (self.haplotypes[1] != 9))

    def get_marker_score(self, ratio=0.9):
        # Ratio determines what percentage more markers the parents need to have than the offspring.
        # If parents and offspring are similar, we want to just run the pop. imputation on the offspring directly.
        # Default is 90%.

        if self.marker_score is None:
            ind_score = np.sum(self.genotypes != 9)

            sire_score = 0
            dam_score = 0

            if self.sire is not None:
                sire_score = self.sire.get_marker_score(ratio)
            if self.dam is not None:
                dam_score = self.dam.get_marker_score(ratio)

            # parent_score = min(sire_score, dam_score)

            (
                self.marker_score,
                self.population_imputation_target,
            ) = self.marker_score_decision_rule(ind_score, sire_score, dam_score, ratio)

        return self.marker_score

    def marker_score_decision_rule_prioritize_individual(
        self, ind_score, sire_score, dam_score, ratio
    ):
        parent_score = min(sire_score, dam_score)

        if ratio * parent_score > ind_score:
            return parent_score, False

        return ind_score, True

    def marker_score_decision_rule_prioritize_parents(
        self, ind_score, sire_score, dam_score, ratio
    ):
        parent_score = min(sire_score, dam_score)

        if parent_score > ratio * ind_score:
            return parent_score, False

        return ind_score, True

    def marker_score_decision_rule_prioritize_parents_diminish(
        self, ind_score, sire_score, dam_score, ratio
    ):
        parent_score = min(sire_score, dam_score)

        if parent_score > 0 and ind_score < 25:
            return max(ind_score, 0.5 * parent_score), False

        if parent_score > ratio * ind_score:
            loss = 0.99
            if ind_score > 100:
                loss = 0.99
            elif ind_score > 50:
                loss = 0.98
            elif ind_score > 25:
                loss = 0.75
            else:
                loss = 0.5

            return max(ind_score, loss * parent_score), False

        return ind_score, True

    # def marker_score_decision_rule_prioritize_balanced(self, ind_score, sire_score, dam_score, ratio):
    #         worst_parent_score = min(sire_score, dam_score)
    #         best_parent_score = min(sire_score, dam_score)

    #         # One of the parents is at a higher density, and the other parent is on roughly the same density.

    #         if ind_score < ratio*best_parent_score and ind_score*ratio < worst_parent_score:
    #             return (sire_score + dam_score)/2, False

    #         return ind_score, True

    def setupIndividual(self):
        nLoci = len(self.genotypes)
        if self.haplotypes is None:
            self.haplotypes = (
                np.full(nLoci, 9, dtype=np.int8),
                np.full(nLoci, 9, dtype=np.int8),
            )

        # self.setPhasingView()
        # self.setPeelingView()

    def setPeelingView(self):
        # Set the
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError(
                "In order to create a jit_Peeling_Individual both the genotypes and haplotypes need to be created."
            )
        nLoci = len(
            self.genotypes
        )  # self.genotypes will always be not None (otherwise error will be raised above).

        if len(self.offspring) > 0:
            self.has_offspring = True
        else:
            self.has_offspring = False

        has_parents = (self.sire is not None) or (self.dam is not None)

        self.peeling_view = jit_Peeling_Individual(
            self.idn,
            self.genotypes,
            self.haplotypes,
            self.has_offspring,
            has_parents,
            nLoci,
            self.map_length,
        )

    def setPhasingView(self):
        # Set the
        if self.genotypes is None or self.haplotypes is None:
            raise ValueError(
                "In order to create a jit_Phasing_Individual both the genotypes and haplotypes need to be created."
            )
        nLoci = len(
            self.genotypes
        )  # self.genotypes will always be not None (otherwise error will be raised above).

        if self.backward_information is None:
            backward = np.full((4, nLoci), 1, dtype=np.float32)
        else:
            backward = self.backward_information
        self.current_haplotypes = (self.haplotypes[0].copy(), self.haplotypes[1].copy())

        self.phasing_view = jit_Phasing_Individual(
            self.idn,
            self.genotypes,
            self.haplotypes,
            backward,
            self.current_haplotypes,
            self.population_imputation_target,
            nLoci,
            self.map_length,
        )

    def reverse_individual(self):
        new_ind = AlphaImputeIndividual(self.idx + "_reverse", self.idn)
        new_ind.genotypes = np.ascontiguousarray(np.flip(self.genotypes))
        new_ind.map_length = self.map_length
        new_ind.population_imputation_target = self.population_imputation_target

        if self.haplotypes is not None:
            new_ind.haplotypes = (
                np.ascontiguousarray(np.flip(self.haplotypes[0])),
                np.ascontiguousarray(np.flip(self.haplotypes[1])),
            )
        else:
            new_ind.setupIndividual()
            Imputation.ind_align(new_ind)

        self.reverse_view = new_ind
        new_ind.reverse_view = self
        return new_ind

    def add_backward_info(self):
        if self.reverse_view is None:
            print(
                "Trying to set backward information, but no reverse_view is availible"
            )
        else:
            self.backward_information = np.flip(
                self.reverse_view.phasing_view.backward, axis=1
            )
            # self.phasing_view.backward[:,:] = np.flip(self.reverse_view.phasing_view.forward, axis = 1) # Flip along loci.

    def clear_reverse_view(self):
        self.reverse_view = None

    def clear_phasing_view(self, keep_current_haplotypes=True):
        self.phasing_view = False
        self.backward_information = None

        if not keep_current_haplotypes:
            self.current_haplotypes = None


spec = OrderedDict()
spec["idn"] = int64
spec["nLoci"] = int64
spec["genotypes"] = int8[:]

# Haplotypes and reads are a tuple of int8 and int64.
spec["haplotypes"] = numba.typeof(
    (np.array([0, 1], dtype=np.int8), np.array([0], dtype=np.int8))
)
spec["current_haplotypes"] = numba.typeof(
    (np.array([0, 1], dtype=np.int8), np.array([0], dtype=np.int8))
)

spec["penetrance"] = float32[:, :]
spec["backward"] = float32[:, :]
spec["map_length"] = float32

spec["own_haplotypes"] = int64[:, :]
spec["has_own_haplotypes"] = boolean
spec["population_imputation_target"] = boolean


@jitclass(spec)
class jit_Phasing_Individual(object):
    """
    This class holds data for phasing a given individual.
    """

    def __init__(
        self,
        idn,
        genotypes,
        haplotypes,
        backward,
        current_haplotypes,
        population_imputation_target,
        nLoci,
        map_length,
    ):
        self.idn = idn
        self.nLoci = nLoci
        self.genotypes = genotypes
        self.haplotypes = haplotypes
        self.current_haplotypes = current_haplotypes

        self.map_length = map_length

        self.penetrance = np.full((4, 1), 1, dtype=np.float32)

        # self.forward = None
        self.backward = backward

        self.own_haplotypes = np.full((0, 0), 0, dtype=np.int64)
        self.has_own_haplotypes = False

        self.population_imputation_target = population_imputation_target

    def set_own_haplotypes(self, haplotypes_in):
        self.own_haplotypes = haplotypes_in
        self.has_own_haplotypes = True

    def setup_penetrance(self):
        self.penetrance = np.full((4, self.nLoci), 1, dtype=np.float32)
        self.setValueFromGenotypes(self.penetrance, 0.01)

    def clear_penetrance(self):
        self.penetrance = np.full((4, 1), 1, dtype=np.float32)

    def setValueFromGenotypes(self, mat, error_rate):
        nLoci = self.nLoci
        mat[:, :] = 1
        for i in range(nLoci):
            g = self.genotypes[i]

            if g == 0:
                mat[0, i] = 1
                mat[1, i] = 0
                mat[2, i] = 0
                mat[3, i] = 0
            if g == 1:
                mat[0, i] = 0
                mat[1, i] = 1
                mat[2, i] = 1
                mat[3, i] = 0

            if g == 2:
                mat[0, i] = 0
                mat[1, i] = 0
                mat[2, i] = 0
                mat[3, i] = 1

            # Handle haplotypes by rulling out genotype states
            if self.haplotypes[0][i] == 0:
                mat[2, i] = 0
                mat[3, i] = 0

            if self.haplotypes[0][i] == 1:
                mat[0, i] = 0
                mat[1, i] = 0

            if self.haplotypes[1][i] == 0:
                mat[1, i] = 0
                mat[3, i] = 0

            if self.haplotypes[1][i] == 1:
                mat[0, i] = 0
                mat[2, i] = 0

            e = error_rate
            count = 0
            for j in range(4):
                count += mat[j, i]
            # if count == 0:
            #     print(g, self.haplotypes[0][i], self.haplotypes[1][i])
            for j in range(4):
                mat[j, i] = mat[j, i] / count * (1 - e) + e / 4


example_phasing_individual = None


def get_example_phasing_individual():
    global example_phasing_individual
    if example_phasing_individual is None:
        example_phasing_individual = jit_Phasing_Individual(
            -1,
            np.array([0, 1], dtype=np.int8),
            (np.array([0, 1], dtype=np.int8), np.array([0, 1], dtype=np.int8)),
            np.full((4, 2), 0, dtype=np.float32),
            (np.array([0, 1], dtype=np.int8), np.array([0, 1], dtype=np.int8)),
            True,
            2,
            1,
        )
    return example_phasing_individual


spec = OrderedDict()

spec["idn"] = int64
spec["nLoci"] = int64
spec["genotypes"] = int8[:]
spec["original_genotypes"] = int8[:]

spec["has_offspring"] = boolean
spec["has_parents"] = boolean
spec["posterior_open"] = boolean
spec["anterior_availible"] = boolean


# Haplotypes and reads are a tuple of int8 and int64.
spec["haplotypes"] = numba.typeof(
    (np.array([0, 1], dtype=np.int8), np.array([0], dtype=np.int8))
)
spec["original_haplotypes"] = numba.typeof(
    (np.array([0, 1], dtype=np.int8), np.array([0], dtype=np.int8))
)

spec["segregation"] = numba.typeof(
    (np.array([0, 1], dtype=np.float32), np.array([0], dtype=np.float32))
)

spec["newPosterior"] = optional(numba.typeof([np.full((4, 100), 0, dtype=np.float32)]))

spec["map_length"] = float32


spec["anterior"] = float32[:, :]
# spec['penetrance'] = float32[:,:]
spec["posterior"] = float32[:, :]
spec["genotypeProbabilities"] = float32[:, :]

spec["currentState"] = int8
spec["currentCutoff"] = float32


@jitclass(spec)
class jit_Peeling_Individual(object):
    """
    This class holds a lot of arrays for peeling individuals and a handful of functions to translate genotypes => probabilities and back again.
    """

    def __init__(
        self, idn, genotypes, haplotypes, has_offspring, has_parents, nLoci, map_length
    ):
        self.nLoci = nLoci
        self.idn = idn
        self.genotypes = genotypes
        self.haplotypes = haplotypes

        self.map_length = map_length

        # Initial value for segregation is .5 to represent uncertainty between haplotype inheritance.
        self.segregation = (
            np.full(nLoci, 0.5, dtype=np.float32),
            np.full(nLoci, 0.5, dtype=np.float32),
        )

        # Create the posterior terms.
        self.has_offspring = has_offspring
        self.has_parents = has_parents

        if self.has_offspring:
            self.original_genotypes = self.genotypes.copy()
            self.original_haplotypes = (
                self.haplotypes[0].copy(),
                self.haplotypes[1].copy(),
            )
            self.anterior_availible = False
            self.posterior_open = False
            self.posterior = np.full((4, nLoci), 1, dtype=np.float32)
            self.anterior = np.full((0, 0), 1, dtype=np.float32)

            # self.anterior = np.full((4, nLoci), 1, dtype = np.float32)
            # self.penetrance = np.full((4, nLoci), 1, dtype = np.float32)
            self.genotypeProbabilities = np.full((4, nLoci), 1, dtype=np.float32)

        else:
            self.original_genotypes = self.genotypes
            self.original_haplotypes = self.haplotypes

            self.posterior = np.full((0, 0), 1, dtype=np.float32)
            self.anterior = np.full((0, 0), 1, dtype=np.float32)
            # self.penetrance = np.full((0, 0), 1, dtype = np.float32)
            self.genotypeProbabilities = np.full((0, 0), 1, dtype=np.float32)

        self.newPosterior = None

        # Current state indicates which genotype combination an individual is set to, and the cutoff value used for calling their genotypes.
        # Missing = -1
        # ALL = 0
        # POSTERIOR = 1
        # PENTRANCE = 2
        # Anterior = 3
        # We only ever use ALL or POSTERIOR.

        self.currentState = -1
        self.currentCutoff = 0

    def check_and_set_state(self, state, cutoff):
        if not self.has_offspring:
            return False

        if self.currentState != state or abs(self.currentCutoff - cutoff) > 0.0001:
            self.currentState = state
            self.currentCutoff = cutoff
            return True
        else:
            return False

    # def fill_in_haplotypes(self):
    #     for i in range(self.nLoci):
    #         if self.genotypes[i] == 0:

    #             if self.haplotypes[0][i] == 9:
    #                 self.haplotypes[0][i] = 0

    #             if self.haplotypes[1][i] == 9:
    #                 self.haplotypes[1][i] = 0

    #         if self.genotypes[i] == 2:

    #             if self.haplotypes[0][i] == 9:
    #                 self.haplotypes[0][i] = 1

    #             if self.haplotypes[1][i] == 9:
    #                 self.haplotypes[1][i] = 1

    def setGenotypesAll(self, cutoff=0.99):
        if self.check_and_set_state(0, cutoff):
            self.setGenotypesFromPeelingData(True, True, True, cutoff)

    def setGenotypesPosterior(self, cutoff=0.99):
        if self.check_and_set_state(1, cutoff):
            self.setGenotypesFromPeelingData(False, True, True, cutoff)

    def setGenotypesPenetrance(self, cutoff=0.99):
        if self.check_and_set_state(2, cutoff):
            self.setGenotypesFromPeelingData(False, True, False, cutoff)

    def setGenotypesAnterior(self, cutoff=0.99):
        if self.check_and_set_state(3, cutoff):
            self.setGenotypesFromPeelingData(True, True, False, cutoff)

    def setAnterior(self, newAnterior):
        self.anterior = newAnterior
        if self.currentState == 0 or self.currentState == 3:
            # i.e. if current state is ALL or ANTERIOR which use the anterior estimate.
            # If the current state is not one of those two, we will re-calculate the genotype probabilities when we set the individual to one of those states.
            self.currentState = -1

        self.anterior_availible = True

    def clearAnterior(self):
        self.anterior = np.full((0, 0), 1, dtype=np.float32)
        self.anterior_availible = False

    def clearPosterior(self):
        if self.has_offspring:
            self.posterior[:, :] = 0
            self.posterior_open = True

    def combinePosterior(self):
        if self.has_offspring and self.newPosterior is not None:
            # Take all the posterior values and add them up.
            if not self.posterior_open:
                print("Trying to add to posterior, but it is not open")
            nPost = len(self.newPosterior)
            for i in range(nPost):
                self.posterior += self.newPosterior[i]
            self.newPosterior = None

    def addPosterior(self, newValues, idn):
        if self.newPosterior is None:
            self.newPosterior = [newValues]
        else:
            self.newPosterior.append(newValues)

    def setPosterior(self):
        if self.has_offspring:
            self.posterior = self.set_posterior_from_scores(self.posterior)
            self.currentState = -1
            self.posterior_open = False

    def set_posterior_from_scores(self, scores):
        nLoci = scores.shape[1]
        posterior = np.full((4, nLoci), 1, dtype=np.float32)
        e = 0.001
        # Maybe could do below in a cleaner fasion, but this is nice and explicit.
        for i in range(nLoci):
            vals = exp_1D_norm(scores[:, i])

            for j in range(4):
                posterior[j, i] = vals[j] * (1 - e) + e / 4
        return posterior

    # def setupProbabilityValues(self, error_rate):
    #     if self.has_offspring: # Only need to do this for founders.
    #         self.setValueFromGenotypes(self.penetrance, error_rate)

    def setValueFromGenotypes(self, mat, error_rate):
        nLoci = self.nLoci
        mat[:, :] = 1
        for i in range(nLoci):
            g = self.original_genotypes[i]

            if g == 0:
                mat[0, i] = 1
                mat[1, i] = 0
                mat[2, i] = 0
                mat[3, i] = 0
            if g == 1:
                mat[0, i] = 0
                mat[1, i] = 1
                mat[2, i] = 1
                mat[3, i] = 0

            if g == 2:
                mat[0, i] = 0
                mat[1, i] = 0
                mat[2, i] = 0
                mat[3, i] = 1

            # Handle haplotypes by rulling out genotype states
            if self.original_haplotypes[0][i] == 0:
                mat[2, i] = 0
                mat[3, i] = 0

            if self.original_haplotypes[0][i] == 1:
                mat[0, i] = 0
                mat[1, i] = 0

            if self.original_haplotypes[1][i] == 0:
                mat[1, i] = 0
                mat[3, i] = 0

            if self.original_haplotypes[1][i] == 1:
                mat[0, i] = 0
                mat[2, i] = 0

            e = error_rate
            count = 0
            for j in range(4):
                count += mat[j, i]

            if count == 0:
                # If phase and genotype disagree, set to missing.
                mat[0, i] = 1
                mat[1, i] = 1
                mat[2, i] = 1
                mat[3, i] = 1
                count = 4
            #     print(g, self.haplotypes[0][i], self.haplotypes[1][i])
            for j in range(4):
                mat[j, i] = mat[j, i] / count * (1 - e) + e / 4

    def setGenotypesFromPeelingData(
        self, useAnterior=False, usePenetrance=False, usePosterior=False, cutoff=0.99
    ):
        # setGenotypesFromPeelingData_ngil(self, useAnterior, usePenetrance, usePosterior, cutoff)
        nLoci = self.nLoci

        self.genotypeProbabilities[:, :] = 1
        finalGenotypes = self.genotypeProbabilities
        if useAnterior and self.has_parents:
            if not self.anterior_availible:
                print("USING ANTERIOR!", self.currentState)

            finalGenotypes *= self.anterior

        if usePosterior and self.has_offspring:
            if self.posterior_open:
                print(
                    "USING POSTERIOR!",
                    self.currentState,
                    self.currentCutoff,
                    self.idn,
                    self.has_parents,
                )
            finalGenotypes *= self.posterior

        if usePenetrance:
            penetrance = np.full((4, nLoci), 1, dtype=np.float32)
            self.setValueFromGenotypes(penetrance, 0.01)
            finalGenotypes *= penetrance

        self.setGenotypesFromGenotypeProbabilities(finalGenotypes, cutoff)

    def setGenotypesFromGenotypeProbabilities(self, finalGenotypes, cutoff):
        nLoci = self.nLoci
        normalize(finalGenotypes)

        # set genotypes/haplotypes from this value.
        for i in range(nLoci):
            # We now set to missing below.
            # self.genotypes[i] = 9
            # self.haplotypes[0][i] = 9
            # self.haplotypes[1][i] = 9

            # Set genotype.
            maxGenotype = 0
            maxVal = finalGenotypes[0, i]

            if finalGenotypes[1, i] + finalGenotypes[2, i] > maxVal:
                maxGenotype = 1
                maxVal = finalGenotypes[1, i] + finalGenotypes[2, i]
            if finalGenotypes[3, i] > maxVal:
                maxGenotype = 2
                maxVal = finalGenotypes[3, i]

            if maxVal > cutoff:
                self.genotypes[i] = maxGenotype
            else:
                self.genotypes[i] = 9

            # Set haplotype.

            hap0 = finalGenotypes[2, i] + finalGenotypes[3, i]
            hap1 = finalGenotypes[1, i] + finalGenotypes[3, i]

            if hap0 > cutoff:
                self.haplotypes[0][i] = 1
            elif hap0 < 1 - cutoff:
                self.haplotypes[0][i] = 0
            else:
                self.haplotypes[0][i] = 9

            if hap1 > cutoff:
                self.haplotypes[1][i] = 1
            elif hap1 < 1 - cutoff:
                self.haplotypes[1][i] = 0
            else:
                self.haplotypes[1][i] = 9


@jit(nopython=True, nogil=True)
def norm_1D(mat):
    total = 0
    for i in range(len(mat)):
        total += mat[i]
    for i in range(len(mat)):
        mat[i] /= total


@jit(nopython=True, nogil=True)
def exp_1D_norm(mat):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1  # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(4):
        if mat[a] > maxVal or maxVal == 1:
            maxVal = mat[a]
    for a in range(4):
        mat[a] -= maxVal

    # Should flag for better numba-ness.
    tmp = np.full(4, 0, dtype=np.float32)
    for a in range(4):
        tmp[a] = np.exp(mat[a])

    norm_1D(tmp)

    return tmp


@jit(nopython=True, nogil=True)
def normalize(values):
    # Normalize values along a single axis.
    for i in range(values.shape[1]):
        count = 0
        for j in range(4):
            count += values[j, i]

        for j in range(4):
            if count != 0:
                values[j, i] /= count
            else:
                values[j, i] = 0.25
