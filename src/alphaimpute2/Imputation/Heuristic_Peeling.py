import numpy as np
from numba import jit, float32, njit
import numba.typed

import concurrent.futures
from itertools import repeat


from ..tinyhouse import ProbMath
from ..tinyhouse.Utils import time_func


# Boiler plate profiler code to make this play nicely with Kernprofiler
if "profile" not in locals():

    def profile(x):
        return x


# Overall pipeline looks something like:
# From top down:
# For the pedigree, calculate segregation, and use it to peel down.
# From bottom up:
# For each parent, peel up offspring.
# For each parent, re-estimate segregation (with peel-up information).

# Error terms:
# Peel down: Assume 0.00001 genotyping/mutation error for anterior terms.
# Peel up: assume 1% in genotypes when converting genotypes => probabilities
# Peel up: assume 1% error in segregation when converting called seg => probabilities.
# Peel up: assume 1% error when going from mate genotype probs => genotype probs.
# Peel up: 1e-8 error rate when marginalizing across mate genotypes (shouldn't need this with the mate genotype probs?)
# Peel up: additional 0.001 error rate when going from posterior -> genotype probabilities.
# Segregation: assumes 1% genotyping error when creating match between individual + parent.


def print_title(text):
    print("")
    print(text)


@profile
@time_func("Heuristic Peeling")
def runHeuristicPeeling(pedigree, args, final_cutoff=0.3):
    # This version of peeling should be run if you are just running pedigree imputation.
    # Either as part of pop_only, or the last step of the combined algorithm.

    # Set up peeling view
    setupHeuristicPeeling(pedigree, args)

    # Run peeling cycle. Cutoffs are for genotype probabilities.
    # Segregation call value is hard-coded to .99
    cutoffs = [0.99] + [args.cutoff for i in range(args.cycles - 1)]
    runPeelingCycles(pedigree, args, cutoffs)

    # Set to best-guess genotypes.
    for ind in pedigree:
        call_genotypes(ind, final_cutoff, args.error)

    # Clear peeling view from all individuals to reduce memory impact.
    for ind in pedigree:
        ind.peeling_view = None


@profile
@time_func("Heuristic Peeling")
def run_integrated_peeling(pedigree, args, final_cutoff=0.3, arrays=None):
    # This version of peeling should be run if you are then going to run population imputation
    # This is the first step of the combined algorithm.
    # It runs peeling cycles, and then splits the population into HD/LD individuals.

    # Set up peeling view
    setupHeuristicPeeling(pedigree, args)

    # Run peeling cycles.
    # Cutoffs are for genotype probabilities.
    # Segregation call value is hard-coded to .99

    cutoffs = [0.99] + [args.cutoff for i in range(args.cycles - 1)]
    runPeelingCycles(pedigree, args, cutoffs)

    for ind in pedigree:
        ind.set_original_genotypes()  # May need to reset individuals
        call_genotypes(ind, final_cutoff, args.error)

    if args.lazy_phasing:
        # Option to build the haplotype library directly from pedigree-phased haplotypes.
        hd_individuals = extract_haplotype_library(pedigree, args, final_cutoff)

    # Split the population into three groups:
    # HD for phasing
    # Individuals for population imputation
    # Individuals for pedigree imputation

    if arrays is not None:
        for array in arrays:
            # Mask markers according to original array density.
            # This seems to improve accuracy in some cases by making sure individuals have a high proportion (>0.9) of non-missing markers on the chip they are on.
            # Possible Improvement: Allow individuals to move arrays before masking.
            mask_array(array)

    for individual in pedigree:
        individual.get_marker_score(args.chip_threshold)  # Set up the marker scores

    if args.lazy_phasing:
        ld_for_pop_imputation = [
            ind for ind in pedigree if ind.population_imputation_target
        ]
        ld_for_ped_imputation = [
            ind for ind in pedigree if not ind.population_imputation_target
        ]

    else:
        hd_individuals = [
            ind for ind in pedigree if np.mean(ind.genotypes != 9) > args.hd_threshold
        ]
        ld_individuals = [
            ind for ind in pedigree if np.mean(ind.genotypes != 9) <= args.hd_threshold
        ]

        ld_for_pop_imputation = [
            ind for ind in ld_individuals if ind.population_imputation_target
        ]
        ld_for_ped_imputation = [
            ind for ind in ld_individuals if not ind.population_imputation_target
        ]

    for ind in ld_for_ped_imputation:
        ind.restore_original_genotypes()

    # Clear peeling view from all individuals to reduce memory impact.
    for ind in pedigree:
        ind.peeling_view = None

    return hd_individuals, ld_for_pop_imputation, ld_for_ped_imputation


def extract_haplotype_library(pedigree, args, final_cutoff):
    # Select individuals with high numbers of phased markers and add them to the pool of hd_individuals.
    # Returns a list of individuals.

    hd_individuals = []
    for ind in pedigree:
        ind.restore_original_genotypes()
        ind.set_original_genotypes()
        call_genotypes(ind, final_cutoff, args.error)

        if ind.percent_phased > 0.95:
            ind.restore_original_genotypes()
            ind.set_original_genotypes()

            call_genotypes(ind, 0.1, args.error)
            clone = ind.copy()
            clone.current_haplotypes = clone.haplotypes
            hd_individuals.append(clone)

            ind.restore_original_genotypes()
            ind.set_original_genotypes()

            call_genotypes(ind, final_cutoff, args.error)

    return hd_individuals


def mask_array(array):
    mask = array.genotypes
    for ind in array.individuals:
        if not ind.original_hd:
            mask_genotypes(ind.genotypes, mask)
            mask_genotypes(ind.haplotypes[0], mask)
            mask_genotypes(ind.haplotypes[1], mask)


@njit
def mask_genotypes(mat, mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            mat[i] = 9


def setupHeuristicPeeling(pedigree, args):
    # Sets the founder anterior values and penetrance value for Heuristic peeling.
    for ind in pedigree:
        ind.setPeelingView()


def call_genotypes(ind, final_cutoff, error_rate):
    # NOTE: THIS WORKS BUT REQUIRES SETTING THESE IN PEDIGREE "PEEL DOWN" ORDER
    # IF NOT, PARENT'S ANTERIOR VALUES MAY NOT BE CORRECTLY SET.
    # FIX: RUN FINAL ROUND OF PEEL DOWN AT THE END.

    if ind.peeling_view.has_offspring:
        if ind.sire is not None and ind.dam is not None:
            # For individuals with offspring, recalculate the anterior value, and then set the genotypes.
            # The posterior is already calculated.

            anterior = getAnterior(
                ind.peeling_view, ind.sire.peeling_view, ind.dam.peeling_view
            )
            ind.peeling_view.setAnterior(anterior)
            ind.peeling_view.setGenotypesAll(final_cutoff)
            ind.peeling_view.clearAnterior()
        else:
            # If no parents, directly set genotypes from the penetrance field.
            ind.peeling_view.setGenotypesAll(final_cutoff)

    else:
        nLoci = len(ind.genotypes)

        penetrance = np.full((4, nLoci), 1, dtype=np.float32)
        ind.peeling_view.setValueFromGenotypes(penetrance, error_rate)

        if ind.sire is not None and ind.dam is not None:
            # Re-calculate parent genotypes with all sources of information.
            ind.sire.peeling_view.setGenotypesAll(ind.sire.peeling_view.currentCutoff)
            ind.dam.peeling_view.setGenotypesAll(ind.dam.peeling_view.currentCutoff)

            # Calculate anterior from the parents.
            anterior = getAnterior(
                ind.peeling_view, ind.sire.peeling_view, ind.dam.peeling_view
            )
            penetrance *= anterior  # Add the penetrance with the anterior. Normalization will happen within the function.

        ind.peeling_view.setGenotypesFromGenotypeProbabilities(penetrance, final_cutoff)


@time_func("Core peeling cycles")
def runPeelingCycles(pedigree, args, cutoffs):
    for cycle, genotype_cutoff in enumerate(cutoffs):
        print_title(f"Imputation cycle {cycle + 1}")
        pedigreePeelDown(pedigree, args, genotype_cutoff)
        pedigreePeelUp(pedigree, args, genotype_cutoff)


@time_func("Peel down")
@profile
def pedigreePeelDown(pedigree, args, cutoff):
    # This function peels down a pedigree; i.e. it finds which regions an individual inherited from their parents, and then fills in the individual's anterior term using that information.
    # To do this peeling, individual's genotypes should be set to poster+penetrance; parent's genotypes should be set to All.
    # Since parents may be shared across families, we set the parents seperately from the rest of the family.
    # We then set the child genotypes, calculate the segregation estimates, and calculate the anterior term on a family by family basis (peel_down_family).

    for generation in pedigree.generations:
        for parent in generation.parents:
            parent.peeling_view.setGenotypesAll(cutoff)

        if args.maxthreads <= 1:
            for family in generation.families:
                peel_down_family(family, cutoff)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.maxthreads
            ) as executor:
                executor.map(peel_down_family, generation.families, repeat(cutoff))


@time_func("Peel up")
@profile
def pedigreePeelUp(pedigree, args, cutoff):
    # This function peels up a pedigree; i.e. it finds which regions an individual inherited from their parents, and then fills in their PARENTS posterior term using that information.
    # To do this peeling, individual's genotypes should be set to poster+penetrance; parent's genotypes should be set to All.
    # Since parents may be shared across families, we set the parents seperately.
    # We then set the child genotypes, calculate the segregation estimates, and calculate the parent's posterior term on a family by family basis (heuristicPeelUp_family).

    for ind in pedigree:
        if ind.peeling_view.has_offspring:
            ind.peeling_view.clearPosterior()

    for generation in reversed(pedigree.generations):
        for parent in generation.parents:
            parent.peeling_view.setGenotypesAll(cutoff)

        if generation.number == 0:
            # Update the posterior value for the ancestors.
            # Note: this is setting the posterior value, not the genotypes.
            for ind in generation.individuals:
                ind.peeling_view.setPosterior()

        if args.maxthreads <= 1:
            for family in generation.families:
                heuristicPeelUp_family(family, cutoff)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.maxthreads
            ) as executor:
                executor.map(
                    heuristicPeelUp_family, generation.families, repeat(cutoff)
                )

        for parent in generation.parents:
            parent.peeling_view.combinePosterior()


############
#
#   Peel Down
#
############


def peel_down_family(family, cutoff):
    sire = family.sire.peeling_view
    dam = family.dam.peeling_view

    # Trying out the new list interface.
    offspring = numba.typed.List()
    for ind in family.offspring:
        offspring.append(ind.peeling_view)

    if len(offspring) > 0:
        peel_down_family_jit(sire, dam, offspring, cutoff)


@jit(nopython=True, nogil=True)
def peel_down_family_jit(sire, dam, offspring, cutoff):
    nOffspring = len(offspring)
    for child in offspring:
        # We don't need to re-set calculate the posterior here, since the posterior value is constant for the peel down pass.
        child.setGenotypesPosterior(cutoff)

    nOffspring = len(offspring)
    for child in offspring:
        setSegregation(child, sire, dam)

    for i in range(nOffspring):
        if offspring[
            i
        ].has_offspring:  # Otherwise we don't really care about the offspring's values.
            newAnterior = getAnterior(offspring[i], sire, dam)
            offspring[i].setAnterior(newAnterior)
            offspring[i].setGenotypesAll(cutoff)
            offspring[i].clearAnterior()


@jit(nopython=True, nogil=True)
def getAnterior(ind, sire, dam):
    # Sets the anterior term based on the parent's genotype probabilities
    nLoci = len(ind.genotypes)

    anterior = np.full((4, nLoci), 0, dtype=np.float32)

    pat_probs = getTransmittedProbs(ind.segregation[0], sire.genotypeProbabilities)
    mat_probs = getTransmittedProbs(ind.segregation[1], dam.genotypeProbabilities)

    # Small error in genotypes. This was to fix some underflow warnings on the multiplications.
    e = 0.00001
    for i in range(nLoci):
        pat = pat_probs[i] * (1 - e) + e / 2
        mat = mat_probs[i] * (1 - e) + e / 2

        anterior[0, i] = (1 - pat) * (1 - mat)
        anterior[1, i] = (1 - pat) * mat
        anterior[2, i] = pat * (1 - mat)
        anterior[3, i] = pat * mat
    return anterior


@jit(nopython=True, nogil=True)
def getTransmittedProbs(seg, genoProbs):
    # For each loci, calculate the probability of the parent transmitting a 1 allele.
    # This will use the child's segregation value, and the parent's genotype probabilities.

    nLoci = len(seg)
    probs = np.full(nLoci, 0.5, np.float32)

    for i in range(nLoci):
        p = genoProbs[:, i]
        p_1 = seg[i] * p[1] + (1 - seg[i]) * p[2] + p[3]
        probs[i] = p_1
    return probs


############
#
#   Peel Up
#
############


def heuristicPeelUp_family(family, cutoff):
    sire = family.sire.peeling_view
    dam = family.dam.peeling_view

    # New peeling interface
    offspring = numba.typed.List()
    for ind in family.offspring:
        offspring.append(ind.peeling_view)

    sire_scores, dam_scores = heuristicPeelUp_family_jit(sire, dam, offspring, cutoff)

    family.sire.peeling_view.addPosterior(sire_scores, family.idn)
    family.dam.peeling_view.addPosterior(dam_scores, family.idn)


@jit(nopython=True, nogil=True)
def heuristicPeelUp_family_jit(sire, dam, offspring, cutoff):
    # Calculate the genotypes for each individual using the Posterior + penetrance.
    # If the individual has offspring, re-calculate the posterior term based on the families already seen in the peel-up operation.

    for child in offspring:
        child.setPosterior()
        child.setGenotypesPosterior(cutoff)

    # Re-estimate the offspring's segregation value
    for child in offspring:
        setSegregation(child, sire, dam)

    # Scores represent the join log genotype probabilities for the sire + dam.
    nLoci = len(sire.genotypes)
    combined_score = np.full((4, 4, nLoci), 0, dtype=np.float32)

    # We peel the child up to both of their parents.
    for child in offspring:
        peelChildToParents(child, combined_score)

    # Calculate individual parental scores by marginalizing over the genotype probabilities of the other parent.
    sire_scores, dam_scores = collapseScoresWithGenotypes(
        combined_score, sire.genotypeProbabilities, dam.genotypeProbabilities
    )
    return sire_scores, dam_scores


@jit(nopython=True, nogil=True)
def peelChildToParents(child, scores):
    # TODO: Test fully inline version of this function.

    nLoci = scores.shape[2]
    for i in range(nLoci):
        if (
            child.genotypes[i] != 9
            or child.haplotypes[0][i] != 9
            or child.haplotypes[1][i] != 9
        ):
            segTensor = getLogSegregationForGenotype(child, i)
            # Summation below was broken out for speed gains in previous versions of numbda
            for j in range(4):
                for k in range(4):
                    scores[j, k, i] += segTensor[j, k]


@jit(nopython=True, nogil=True)
def getLogSegregationForGenotype(child, i):
    # Basically we want to be able to go from seg[0] + seg[1] + genotype + hap[0] + hap[1] => joint parental genotype.

    seg_threshold = 0.99

    seg0 = convert_seg_to_int(child.segregation[0][i], seg_threshold)
    seg1 = convert_seg_to_int(child.segregation[1][i], seg_threshold)

    hap0 = child.haplotypes[0][i]
    hap1 = child.haplotypes[1][i]

    geno = child.genotypes[i]
    return logGenotypeSegregationTensor[seg0, seg1, hap0, hap1, geno]


@jit(nopython=True, nogil=True)
def convert_seg_to_int(val, threshold):
    if val < 1 - threshold:
        return 0
    if val > threshold:
        return 1
    return 9


@jit(nopython=True, nogil=True)
def collapseScoresWithGenotypes(scores, sire_genotype_probs, dam_genotype_probs):
    # Takes in log-scores; outputs log-posterior for sire and dam.
    nLoci = dam_genotype_probs.shape[-1]

    sire_score = np.full((4, nLoci), 0, dtype=np.float32)
    dam_score = np.full((4, nLoci), 0, dtype=np.float32)

    e = 0.001
    values = np.full((4, 4), 0, dtype=np.float32)

    for i in range(nLoci):
        # Convert log-scores to scores
        exp_2D_norm(scores[:, :, i], values)

        # Calculate sire posterior term
        for j in range(4):
            for k in range(4):
                sire_score[j, i] += values[j, k] * (
                    dam_genotype_probs[k, i] * (1 - e) + e / 4
                )

        norm_1D(sire_score[:, i])
        sire_score[:, i] += 1e-8

        # Calculate dam posterior term
        for j in range(4):
            for k in range(4):
                dam_score[j, i] += values[k, j] * (
                    sire_genotype_probs[k, i] * (1 - e) + e / 4
                )

        norm_1D(dam_score[:, i])
        dam_score[:, i] += 1e-8

    return np.log(sire_score), np.log(dam_score)


############
#
#   Estimate Segregation
#
############


@jit(nopython=True, nogil=True)
def setSegregation(ind, sire, dam):
    nLoci = len(ind.genotypes)

    pointEstimates = np.full((4, nLoci), 1, dtype=np.float32)
    fillPointEstimates(pointEstimates, ind, sire, dam)

    # Runs a forward backward algorithm on the pointEstimates
    smoothedEstimates = smoothPointSeg(
        pointEstimates, 1.0 / nLoci * ind.map_length
    )  # This is where different map lengths could be added.

    # Then set the segregation values for the individual.
    ind.segregation[0][:] = smoothedEstimates[2, :] + smoothedEstimates[3, :]
    ind.segregation[1][:] = smoothedEstimates[1, :] + smoothedEstimates[3, :]


@jit(nopython=True, nogil=True)
def fillPointEstimates(pointEstimates, ind, sire, dam):
    # Calculate probability of each segregation state conditional on parent's genotype state and own genotypes.
    nLoci = pointEstimates.shape[1]
    e = 0.01  # Assume 1% genotyping error.
    for i in range(nLoci):
        # Let's do sire side.
        # I'm going to assume we've already peeled down.

        sirehap0 = sire.haplotypes[0][i]
        sirehap1 = sire.haplotypes[1][i]
        damhap0 = dam.haplotypes[0][i]
        damhap1 = dam.haplotypes[1][i]

        # There's an extra edge case where both the child is heterozygous, unphased, and both the parent's haplotypes are phased.
        if (
            ind.genotypes[i] == 1
            and ind.haplotypes[0][i] == 9
            and ind.haplotypes[0][i] == 9
        ):
            if sirehap0 != 9 and sirehap1 != 9 and damhap0 != 9 and damhap1 != 9:
                # This is ugly, but don't have a better solution.

                if sirehap0 + damhap0 == 1:
                    pointEstimates[0, i] *= 1 - e
                else:
                    pointEstimates[0, i] *= e

                if sirehap0 + damhap1 == 1:
                    pointEstimates[1, i] *= 1 - e
                else:
                    pointEstimates[1, i] *= e

                if sirehap1 + damhap0 == 1:
                    pointEstimates[2, i] *= 1 - e
                else:
                    pointEstimates[2, i] *= e

                if sirehap1 + damhap1 == 1:
                    pointEstimates[3, i] *= 1 - e
                else:
                    pointEstimates[3, i] *= e

        if ind.haplotypes[0][i] != 9:
            indhap = ind.haplotypes[0][i]

            # If both parental haplotypes are non-missing, and
            # not equal to each other, then this is an informative marker.
            if sirehap0 != 9 and sirehap1 != 9 and sirehap0 != sirehap1:
                if indhap == sirehap0:
                    pointEstimates[0, i] *= 1 - e
                    pointEstimates[1, i] *= 1 - e
                    pointEstimates[2, i] *= e
                    pointEstimates[3, i] *= e

                if indhap == sirehap1:
                    pointEstimates[0, i] *= e
                    pointEstimates[1, i] *= e
                    pointEstimates[2, i] *= 1 - e
                    pointEstimates[3, i] *= 1 - e

        if ind.haplotypes[1][i] != 9:
            indhap = ind.haplotypes[1][i]

            if damhap0 != 9 and damhap1 != 9 and damhap0 != damhap1:
                if indhap == damhap0:
                    pointEstimates[0, i] *= 1 - e
                    pointEstimates[1, i] *= e
                    pointEstimates[2, i] *= 1 - e
                    pointEstimates[3, i] *= e

                if indhap == damhap1:
                    pointEstimates[0, i] *= e
                    pointEstimates[1, i] *= 1 - e
                    pointEstimates[2, i] *= e
                    pointEstimates[3, i] *= 1 - e


@jit(
    nopython=True,
    nogil=True,
    locals={"e": float32, "e2": float32, "e1e": float32, "e2i": float32},
)
def smoothPointSeg(pointSeg, transmission):
    nLoci = pointSeg.shape[1]

    # Seg is the output, and is a copy of pointseg.
    seg = np.full(pointSeg.shape, 0.25, dtype=np.float32)
    for i in range(nLoci):
        for j in range(4):
            seg[j, i] = pointSeg[j, i]

    # Variables.
    tmp = np.full(4, 0, dtype=np.float32)
    new = np.full(4, 0, dtype=np.float32)
    prev = np.full(4, 0.25, dtype=np.float32)

    # Transmission constants.
    e = transmission
    e2 = e**2
    e1e = e * (1 - e)
    e2i = (1.0 - e) ** 2

    for i in range(1, nLoci):
        # Combine previous estimate with previous pointseg and then transmit forward.
        for j in range(4):
            tmp[j] = prev[j] * pointSeg[j, i - 1]

        norm_1D(tmp)

        # Father/Mother; Paternal/Maternal
        # !                  fm  fm  fm  fm
        # !segregationOrder: pp, pm, mp, mm

        new[0] = e2 * tmp[3] + e1e * (tmp[1] + tmp[2]) + e2i * tmp[0]
        new[1] = e2 * tmp[2] + e1e * (tmp[0] + tmp[3]) + e2i * tmp[1]
        new[2] = e2 * tmp[1] + e1e * (tmp[0] + tmp[3]) + e2i * tmp[2]
        new[3] = e2 * tmp[0] + e1e * (tmp[1] + tmp[2]) + e2i * tmp[3]

        for j in range(4):
            seg[j, i] *= new[j]
        prev = new

    prev = np.full((4), 0.25, dtype=np.float32)
    for i in range(
        nLoci - 2, -1, -1
    ):  # zero indexed then minus one since we skip the boundary.
        for j in range(4):
            tmp[j] = prev[j] * pointSeg[j, i + 1]

        norm_1D(tmp)

        new[0] = e2 * tmp[3] + e1e * (tmp[1] + tmp[2]) + e2i * tmp[0]
        new[1] = e2 * tmp[2] + e1e * (tmp[0] + tmp[3]) + e2i * tmp[1]
        new[2] = e2 * tmp[1] + e1e * (tmp[0] + tmp[3]) + e2i * tmp[2]
        new[3] = e2 * tmp[0] + e1e * (tmp[1] + tmp[2]) + e2i * tmp[3]

        for j in range(4):
            seg[j, i] *= new[j]
        prev = new

    for i in range(nLoci):
        norm_1D(seg[:, i])

    return seg


############
#
#  Numba matrix functions
#
###########


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
def exp_2D_norm(mat, output):
    # Matrix is 4x4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix (in place).
    # Question: Does explicit dimensionality help or hinder?
    # Question: Could we also make this work for nLoci as well?
    # i.e. can we stick these all somewhere together and not replicate?
    maxVal = 1  # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(4):
        for b in range(4):
            if mat[a, b] > maxVal or maxVal == 1:
                maxVal = mat[a, b]

    # Should flag for better numba-ness.
    for a in range(4):
        for b in range(4):
            output[a, b] = np.exp(mat[a, b] - maxVal)

    # Normalize.
    score = 0
    for a in range(4):
        for b in range(4):
            score += output[a, b]
    for a in range(4):
        for b in range(4):
            output[a, b] /= score


############
#
#   STATIC METHODS + GLOBALS
#
############

# This sets up how you turn haplotypes + genotypes into genotype probabilities.
# Indexing is hap0, hap1, geno, GenotypeProbabilities.


def generateGenoProbs():
    global geno_probs
    error = 0.01
    geno_probs[9, 9, 9] = (
        np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[9, 9, 1] = (
        np.array([0, 0.5, 0.5, 0], dtype=np.float32) * (1 - error) + error / 4
    )

    geno_probs[0, 9, 9] = (
        np.array([0.5, 0.5, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[1, 9, 9] = (
        np.array([0, 0, 0.5, 0.5], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[9, 0, 9] = (
        np.array([0.5, 0, 0.5, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[9, 1, 9] = (
        np.array([0, 0.5, 0, 0.5], dtype=np.float32) * (1 - error) + error / 4
    )

    geno_probs[1, 0, 1] = (
        np.array([0, 0, 1, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[0, 1, 1] = (
        np.array([0, 1, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[0, 0, 0] = (
        np.array([1, 0, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    geno_probs[1, 1, 2] = (
        np.array([0, 0, 0, 1], dtype=np.float32) * (1 - error) + error / 4
    )


geno_probs = np.full(
    (10, 10, 10, 4), 0.25, dtype=np.float32
)  # Because 9 indexing for missing.
generateGenoProbs()

# Now generate a segregation tensor that goes from genotypes -> phasedGenotypes + error -> probs on parents.
# logGenotypeSegregationTensor = np.log(genotypeSegregationTensor)


def generateLogGenotypeSegregationTensor():
    # Assume 1% genotyping error rate and 1% segregation error rate.

    error = 0.01
    global logGenotypeSegregationTensor
    segregationTensor = (
        ProbMath.generateSegregation()
    )  # This will be sire, dam, offspring.

    # segregation probabilities for each possible segregation value.
    seg_probs = np.full((10, 10, 4), 0.25, dtype=np.float32)
    seg_probs[0, 0] = np.array([1, 0, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    seg_probs[0, 1] = np.array([0, 1, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    seg_probs[1, 0] = np.array([0, 0, 1, 0], dtype=np.float32) * (1 - error) + error / 4
    seg_probs[1, 1] = np.array([0, 0, 0, 1], dtype=np.float32) * (1 - error) + error / 4

    seg_probs[0, 9] = (
        np.array([0.5, 0.5, 0, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    seg_probs[1, 9] = (
        np.array([0, 0, 0.5, 0.5], dtype=np.float32) * (1 - error) + error / 4
    )

    seg_probs[9, 0] = (
        np.array([0.5, 0, 0.5, 0], dtype=np.float32) * (1 - error) + error / 4
    )
    seg_probs[9, 1] = (
        np.array([0, 0.5, 0, 0.5], dtype=np.float32) * (1 - error) + error / 4
    )

    seg_probs[9, 9] = (
        np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32) * (1 - error) + error / 4
    )

    genotypeSegregationTensor = np.full(
        (10, 10, 10, 10, 10, 4, 4), 0.25, dtype=np.float32
    )

    for seg0 in [0, 1, 9]:
        for seg1 in [0, 1, 9]:
            for hap0 in [0, 1, 9]:
                for hap1 in [0, 1, 9]:
                    for geno in [0, 1, 2, 9]:
                        genotypes = geno_probs[hap0, hap1, geno]
                        segregation = seg_probs[seg0, seg1]

                        genotypeSegregationTensor[
                            seg0, seg1, hap0, hap1, geno, :, :
                        ] = np.einsum(
                            "abcd, c, d -> ab",
                            segregationTensor,
                            genotypes,
                            segregation,
                        )
                        logGenotypeSegregationTensor = np.log(genotypeSegregationTensor)


logGenotypeSegregationTensor = None
generateLogGenotypeSegregationTensor()
