rm(list = ls())
# install.packages(pkg = "AlphaSimR")
library(AlphaSimR)

# ---- Two (or more) sub-populations A and B which cross in later generations ----

parameters <- read.table("../simulation_parameters.txt")
nparams <- nrow(parameters)
for (parameter in (1:nparams)) {
  eval(parse(text = paste0(parameters$V1[parameter], "<-", parameters$V2[parameter])))
}

nIndPerGen <- nInd / nGen

nLociAllPerChr <- floor(nLociAll / nChr)

founderGenomes <- runMacs(nInd = nIndPerGen, nChr = nChr, segSites = nLociAllPerChr, split = 1000)
SP <- SimParam$new(founderPop = founderGenomes)
SP$setSexes("yes_rand")
SP$setTrackPed(isTrackPed = TRUE)
SP$setTrackRec(isTrackRec = TRUE)
varG <- matrix(data = c( 1.0, -0.3,
                         -0.3,  1.0), byrow = TRUE, nrow = 2)
SP$addTraitA(nQtlPerChr = 1000, mean = c(0, 0), var = diag(varG), corA = varG)
varE <- matrix(data = c(2.0, 0.0,
                        0.0, 2.0), byrow = TRUE, nrow = 2)

collectData <- function(pop, data = NULL, population, generation) {
  remove <- FALSE
  if (is.null(data)) {
    remove <- TRUE
    data <- vector(mode = "list", length = 3)
    names(data) <- c("pedigree", "Haplotype", "Genotype")
    data$pedigree <- data.frame(id = NA, population = NA, generation = NA,
                                mid = NA, fid = NA)
    data$Haplotype <- matrix(data = NA, ncol = sum(pop@nLoci))
    data$Genotype <- matrix(data = NA, ncol = sum(pop@nLoci))
  }
  data$pedigree <- rbind(data$pedigree,
                         data.frame(id = pop@id,
                                    population = population,
                                    generation = generation,
                                    mid = pop@mother,
                                    fid = pop@father
                                    ))
  data$Haplotype <- rbind(data$Haplotype,
                         pullSegSiteHaplo(pop = pop))
  data$Genotype <- rbind(data$Genotype,
                        pullSegSiteGeno(pop = pop))
  if (remove) {
    data$pedigree <- data$pedigree[-1, ]
    data$Haplotype <- data$Haplotype[-1, ]
    data$Genotype <- data$Genotype[-1, ]
  }
  return(data)
}

# Founder population & split
# Number of individuals for each pop (A, B, and AB)
subPopEarlyGen <- nIndPerGen*0.5
subPopAperGen <- nIndPerGen*0.375
subPopBperGen <- nIndPerGen*0.375
subPopABperGen <- nIndPerGen*0.25
nIndFoundersPerSubPop <- nIndPerGen*0.5
nSelInd <- nIndPerGen*0.05

founders <- newPop(rawPop = founderGenomes)
founders <- setPheno(pop = founders, varE = diag(varE))
popA <- founders[1:nIndFoundersPerSubPop]
popB <- founders[(nIndFoundersPerSubPop+1):nIndPerGen]
data <- collectData(pop = popA, data = NULL, population = "A", generation = 0)
data <- collectData(pop = popB, data = data, population = "B", generation = 0)

# Select on each trait and keep the populations separate
for (generation in 1:nGen) {
  parentsA <- selectInd(pop = popA, nInd = nSelInd, trait = 1)
  parentsB <- selectInd(pop = popB, nInd = nSelInd, trait = 2)
  if (generation == nGen){
    popA <- randCross(pop = parentsA, nCrosses = subPopAperGen)
    popB <- randCross(pop = parentsB, nCrosses = subPopBperGen)
  } else {
    popA <- randCross(pop = parentsA, nCrosses = subPopEarlyGen)
    popB <- randCross(pop = parentsB, nCrosses = subPopEarlyGen)
  }
  
  popA <- setPheno(pop = popA, varE = diag(varE))
  popB <- setPheno(pop = popB, varE = diag(varE))
  data <- collectData(pop = popA, data = data, population = "A", generation = generation)
  data <- collectData(pop = popB, data = data, population = "B", generation = generation)
}

# Continued selection on each trait in each separate population,
# but add also continually admixed population selected on an index
popAB <- randCross(pop = c(parentsA, parentsB), nCrosses = subPopABperGen)
popAB <- setPheno(pop = popAB, varE = diag(varE))
data <- collectData(pop = popAB, data = data, population = "AB", generation = generation)
economicWeights <- c(1, 1)
selIndexWeights <- smithHazel(econWt = economicWeights, varG = varG, varP = varG + varE)
for (generation in (nGen+1):(nGen*2)) {
  parentsA <- selectInd(pop = popA, nInd = nSelInd, trait = 1)
  parentsB <- selectInd(pop = popB, nInd = nSelInd, trait = 2)
  parentsAB <- selectInd(pop = popAB, nInd = nSelInd*0.6, trait = selIndex, scale = TRUE, b = selIndexWeights)
  parentsA4AB <- selectInd(pop = popA, nInd = nSelInd*0.2, trait = selIndex, scale = TRUE, b = selIndexWeights)
  parentsB4AB <- selectInd(pop = popB, nInd = nSelInd*0.2, trait = selIndex, scale = TRUE, b = selIndexWeights)
  parentsAB <- c(parentsAB, parentsA4AB, parentsB4AB)
  popA <- randCross(pop = parentsA, nCrosses = subPopAperGen)
  popB <- randCross(pop = parentsB, nCrosses = subPopBperGen)
  popAB <- randCross(pop = parentsAB, nCrosses = subPopABperGen)
  popA <- setPheno(pop = popA, varE = diag(varE))
  popB <- setPheno(pop = popB, varE = diag(varE))
  popAB <- setPheno(pop = popAB, varE = diag(varE))
  data <- collectData(pop = popA,  data = data, population = "A",  generation = generation)
  data <- collectData(pop = popB,  data = data, population = "B",  generation = generation)
  data <- collectData(pop = popAB, data = data, population = "AB", generation = generation)
}

data$pedigree$population <- factor(data$pedigree$population, levels = c("A", "B", "AB"))
summary(data$pedigree$population)

# Get pedigree and assign metafounders
pedigree <- data.frame(data$pedigree$id, data$pedigree$mid, data$pedigree$fid, data$pedigree$population, data$pedigree$generation)
colnames(pedigree) <- c("id", "mid", "fid", "population", "generation")
# Take nInd (1000) individuals across nGen (five) generations.
pedStart <- nIndPerGen*2
pedEnd <- pedStart + nInd
pedigree <- pedigree[c((pedStart + 1):pedEnd),]
pedigree$mid[c(1:nIndFoundersPerSubPop)] <- "MF_1"
pedigree$mid[c((nIndFoundersPerSubPop + 1):nIndPerGen)] <- "MF_2"
pedigree$fid[c(1:nIndFoundersPerSubPop)] <- "MF_1"
pedigree$fid[c((nIndFoundersPerSubPop + 1):nIndPerGen)] <- "MF_2"

pedigree <- pedigree[c(-4, -5)]

# Get the genotypes (true) and observed (i.e with some missing)
genotypes <- data.frame(data$Genotype)
genotypes <- genotypes[(pedStart + 1):pedEnd, ]
genotypes_obsv <- genotypes

# Define sample sizes and corresponding offsets
nUnknown <- c(0.7, 0.7, 0.7, 0.7, 0.4, 0.4) * nIndFoundersPerSubPop
offsets <- c(0, 1, 2, 3, 4, 5) * nIndFoundersPerSubPop

for (i in seq_along(nUnknown)){
  randFounder <- sort(sample.int(nIndFoundersPerSubPop, nUnknown[i])) + offsets[i]
  genotypes_obsv[randFounder, 1:nLociAllPerChr] <- "9"
}

# Estimate the alternative allele frequencies.
founders_genotypes <- genotypes[c(1:nIndFoundersPerSubPop),]
alt_allele_prob_A <- colSums(founders_genotypes)
alt_allele_prob_A <- alt_allele_prob_A/(2*nrow(founders_genotypes))

founders_genotypes <- genotypes[c((nIndFoundersPerSubPop + 1):nIndPerGen),]
alt_allele_prob_B <- colSums(founders_genotypes)
alt_allele_prob_B <- alt_allele_prob_B/(2*nrow(founders_genotypes))

alt_allele_prob_input_MF <- data.frame(matrix(nrow = 2, ncol = (nLociAllPerChr + 1)))
alt_allele_prob_input_MF[1] <- c("MF_1", "MF_2")
alt_allele_prob_input_MF[1,2:(nLociAllPerChr + 1)] <- alt_allele_prob_A
alt_allele_prob_input_MF[2,2:(nLociAllPerChr + 1)] <- alt_allele_prob_B

# Singular alternative allele frequency is not used in testing, but calculated for comparison.
founders_genotypes <- genotypes[c(1:nIndPerGen),]
alt_allele_prob <- colSums(founders_genotypes)
alt_allele_prob <- alt_allele_prob/(2*nrow(founders_genotypes))

alt_allele_prob_input_noMF <- data.frame(matrix(nrow = 1, ncol = (nLociAllPerChr + 1)))
alt_allele_prob_input_noMF[1] <- "MF_1"
alt_allele_prob_input_noMF[1,2:(nLociAllPerChr + 1)] <- alt_allele_prob

# Haplotypes
# Get haplotypes for phased and unphased genotype probabilities
haplotypes <- data.frame(data$Haplotype)
haplotypes <- data$Haplotype[c((1+pedStart*2):(pedEnd*2)),]

for (ind in 1:nInd) {
  maternal <- haplotypes[ind * 2 - 1, ]
  haplotypes[ind * 2 - 1, ] <- haplotypes[ind * 2, ]
  haplotypes[ind * 2, ] <- maternal
}

# ----- Phased genotypes probability------

phasedGenotypes <- matrix(data = 0, nrow = nInd * 4, ncol = nLociAll + 1)
for (ind in (1:nInd)) {
  for (locus in (1:nLociAll)) {
    currentGeno <- haplotypes[((ind - 1) * 2 + 1):(ind * 2), locus]
    if (all(currentGeno == c(0, 0)) == TRUE) {
      currentPhasedGeno <- c(1, 0, 0, 0)
    } else if (all(currentGeno == c(0, 1)) == TRUE) {
      currentPhasedGeno<- c(0, 1, 0, 0)
    } else if (all(currentGeno == c(1, 0)) == TRUE) {
      currentPhasedGeno <- c(0, 0, 1, 0)
    } else {
      currentPhasedGeno <- c(0, 0, 0, 1)
    }
    phasedGenotypes[((ind - 1) * 4 + 1):(ind * 4), locus + 1] <- currentPhasedGeno
  }
  phasedGenotypes[((ind - 1) * 4 + 1):(ind * 4), 1] <- ind
}

# ----- (Unphased) genotype probability -----

UnphasedGenotypes <- matrix(data = 0, nrow = nInd * 3, ncol = nLociAll + 1)
for (ind in (1:nInd)) {
  for (locus in (1:nLociAll)) {
    currentGeno <- haplotypes[((ind - 1) * 2 + 1):(ind * 2), locus]
    if (all(currentGeno == c(0, 0)) == TRUE) {
      currentUnphasedGeno <- c(1, 0, 0)
    } else if (all(currentGeno == c(0, 1)) == TRUE) {
      currentUnphasedGeno<- c(0, 1, 0)
    } else if (all(currentGeno == c(1, 0)) == TRUE) {
      currentUnphasedGeno <- c(0, 1, 0)
    } else {
      currentUnphasedGeno <- c(0, 0, 1)
    }
    UnphasedGenotypes[((ind - 1) * 3 + 1):(ind * 3), locus + 1] <- currentUnphasedGeno
  }
  UnphasedGenotypes[((ind - 1) * 3 + 1):(ind * 3), 1] <- ind
}

# To match the ids
phasedGenotypes[,1] <- phasedGenotypes[,1] + pedStart
UnphasedGenotypes[,1] <- UnphasedGenotypes[,1] + pedStart

# Save the results/data for testing
write.table(x = pedigree, file = "metafounder_ped_file.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = genotypes_obsv, file = "metafounder_geno_file.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(x = genotypes, file = "true-metafounder_dosage.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(x = UnphasedGenotypes, file = "true-metafounder_geno_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = phasedGenotypes, file = "true-metafounder_phased_geno_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = alt_allele_prob_input_MF, file = "metafounder_alt_allele_prob_file.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)


