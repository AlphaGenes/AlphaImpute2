# ----- Setup -----

rm(list = ls())
# install.packages(pkg = "AlphaSimR")
library(AlphaSimR)

# ----- Functions -----

# generate errors in the genotype input
# geno - matrix, genotypes (ind x loc) coded as 0, 1, 2, or 9 (missing)
# error - numeric, probability of observing an error (single value or
#         multiple to add variation across loci)
generateGenoErr <- function(geno, error) {
  nLoci <- ncol(geno)
  for (ind in 1:nInd){
    for (locus in 1:nLoci){
      if (geno[ind, locus] != 9 && rbinom(n = 1, size = 1, prob = error) == 1) {
        if (geno[ind, locus] == 0) {
          geno[ind, locus] <- sample(c(1, 2), size = 1)
        } else if (geno[ind, locus] == 1) {
          geno[ind, locus] <- sample(c(0, 2), size = 1)
        } else {
          geno[ind, locus] <- sample(c(0, 1), size = 1)
        }
      }
    }
  }
  return(geno)
}

# simulateSeqReads
url <- "https://gist.githubusercontent.com/gregorgorjanc/e7ca7a02ed59242573c12a890cf8c871/raw/e88ad2a3036f3385a0724d9d92a7cf5012ee0f86/simulateSeqReads.R"
source(file = url)

# ----- Simulation parameters -----

parameters <- read.table("../simulation_parameters.txt")
nparams <- nrow(parameters)
for (parameter in (1:nparams)) {
  eval(parse(text = paste0(parameters$V1[parameter], "<-", parameters$V2[parameter])))
}

nIndPerGen <- nInd / nGen

nLociAllPerChr <- floor(nLociAll / nChr)
nLociHDPerChr <- nLociHD / nChr
nLociLDPerChr <- nLociLD / nChr

# ----- SimParam and base population -----

founderGenomes <- runMacs(nInd = nIndPerGen, nChr = nChr, segSites = nLociAllPerChr)
SP <- SimParam$new(founderGenomes)
SP$setSexes("yes_rand")
SP$setTrackPed(TRUE)
SP$setTrackRec(TRUE)
SP$addSnpChip(nSnpPerChr = nLociHDPerChr, name = "HD")

# Create LD chip
by <- nLociHDPerChr / nLociLDPerChr
tmp <- seq(from = by / 2, to = nLociHDPerChr, by = by)
markersLD <- c(outer(X = 1:nChr, Y = tmp, FUN = paste, sep = "_"))
tmp <- c(outer(X = 1:nChr, Y = 1:nLociHDPerChr, FUN = paste, sep = "_"))
markersHD_without_LD <- tmp[!tmp %in% markersLD]
# SP$addSnpChipByName(markers = markersLD, name = "LD") --> we will subset HD later for some individuals

basePop <- newPop(founderGenomes)

# ----- Generate individuals across generations -----

currentPop <- basePop
allPop <- currentPop
Parents <- c()
for (generation in 1:(nGen - 1)) {
  newParents <- selectInd(currentPop, nInd = nParents, use = "rand")
  Parents <- c(Parents, newParents@id)
  currentPop <- randCross(newParents, nCrosses = nIndPerGen)
  allPop <- c(allPop, currentPop)
}

# ----- Collect/Generate data & Calculate realised rates -----

# ----- Pedigree -----

pedigree <- getPed(pop = allPop)
pedigree <- pedigree[, c("id", "father", "mother")]


# ----- Haplotypes ----

haplotypes <- pullSnpHaplo(pop = allPop, snpChip = "HD")
# flip maternal and paternal haplotype (AlphaSimR puts maternal first, AlphaPeel paternal)
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

# ----- Genotypes -----

genotypes <- pullSnpGeno(pop = allPop, snpChip = "HD")
genotypesObs <- genotypes
for (ind in 1:nInd) {
  if (!ind %in% Parents) {
    genotypesObs[ind, markersHD_without_LD] <- 9
  }
}
genotypesObs_w_error <- generateGenoErr(geno = genotypesObs, error = genoError)

# ----- Realised allele frequency in the base population -----

alt_allele_prob <- matrix(data = colMeans(x = genotypes[1:nIndPerGen, ]) / 2,
              nrow = nLociAll, ncol = 1)
summary(alt_allele_prob[, 1])

# ----- Realised genotype error rate -----

geno_error <- matrix(0, nrow = nLociAll, ncol = 1)
for (locus in 1:nLociAll) {
  total <- 0
  nerrors <- 0
  for (ind in 1:nInd) {
    if (genotypesObs_w_error[ind, locus] != 9) {
      total <- total + 1
      if (genotypesObs_w_error[ind, locus] != genotypes[ind, locus]) {
        nerrors <- nerrors + 1
      }
    }
  }
  geno_error[locus, ] <- nerrors / total
}
genoError
summary(geno_error[, 1])

# ----- Sequence reads -----

sequenceReads <- simulateSeqReads(genotypes, depth = seqDepth, error = seqError)
idColumn <- matrix(c(rep(1:nInd, each = 2)), nrow = nInd * 2, ncol = 1)
sequenceReads <- cbind(idColumn, sequenceReads)

# ----- Realised sequence error rate -----

seq_error <- matrix(data = 0, nrow = nLociAll, ncol = 1)
for (locus in 1:nLociAll) {
  startInd <- 1
  endInd <- 2
  nSeqReadError <- 0
  nSeqReadTotal <- sum(sequenceReads[, locus])
  for (ind in 1:nInd) {
    nRef <- sequenceReads[startInd, locus]
    nAlt <- sequenceReads[endInd, locus]
    if (genotypes[ind, locus] == 0) {
      nSeqReadError <- nSeqReadError + nAlt
    } else if (genotypes[ind, locus] == 2) {
      nSeqReadError <- nSeqReadError + nRef
    }
    startInd <- endInd + 1
    endInd <- startInd + 1
  }
  seq_error[locus, 1] <- nSeqReadError / nSeqReadTotal
}
seqError
summary(seq_error[, 1])

# ----- Recombination events ------

# Needed for the calculation of segregation and the recombination rate
# recHist[[individual]][[chromosome]][[haplotype]] 
# column 1 gives the parental haplotype, column 2 gives the start loci
recHist <- SP$recHist

# ----- Realised segregation ------

segregation <- matrix(data = 0, nrow = nInd * 4, ncol = nLociAll + 1)
segregation[1:(4 * nIndPerGen), 1] <- rep(x = 1:nIndPerGen, each = 4)
segregation[1:(4 * nIndPerGen), 2:(nLociAll + 1)] <- 
  matrix(data = rep(0.25, times = nLociAll * 4 * nIndPerGen), 
         nrow = 4 * nIndPerGen, ncol = nLociAll)

for (ind in (nIndPerGen + 1):nInd) {
  indRecHist <- recHist[[ind]][[1]]
  
  nMaternalComb <- nrow(indRecHist[[1]])
  maternalPattern <- matrix(0, nrow = 4, ncol = nLociAll)
  for (comb in (1:nMaternalComb)) {
    start <- indRecHist[[1]][comb, 2]
    if (indRecHist[[1]][comb, 1] == 1) {
      # maternal haplotype is inherited from the individual's grandmaternal
      pattern <- c(0, 1, 0, 1)
    } else {
      # maternal haplotype is inherited from the individual's grandpaternal
      pattern <- c(1, 0, 1, 0)
    }
    maternalPattern[1:4, start:nLociAll] <- 
      matrix(data = rep(pattern, times = nLociAll - start + 1), 
             nrow = 4, ncol = nLociAll - start + 1)
  }
  
  nPaternalComb <- nrow(indRecHist[[2]])
  paternalPattern <- matrix(0, nrow = 4, ncol = nLociAll)
  for (comb in (1:nPaternalComb)) {
    start <- indRecHist[[2]][comb, 2]
    if (indRecHist[[2]][comb, 1] == 1) {
      # paternal haplotype is inherited from the individual's grandmaternal
      pattern <- c(0, 0, 1, 1)
    } else {
      # paternal haplotype is inherited from the individual's grandpaternal
      pattern <- c(1, 1, 0, 0)
    }
    paternalPattern[1:4, start:nLociAll] <- 
      matrix(data = rep(pattern, times = nLociAll - start + 1),
             nrow = 4, ncol = nLociAll - start + 1)
  }
  
  startRow <- 4 * (ind - 1) + 1
  endRow <- 4 * ind
  segregation[startRow:endRow, 2:(nLociAll + 1)] <- maternalPattern * paternalPattern
}

# ----- Realised recombination rate -----

rec_count <- matrix(data = 0, nrow = nLociAll, ncol = 1)
for (ind in ((nIndPerGen + 1):nInd)) {
  for (chr in (1:nChr)) {
    indRecHist <- recHist[[ind]][[chr]]
    for (haplo in (1:2)) {
      nComb <- nrow(indRecHist[[haplo]])
      if (nComb > 1) {
        for (comb in (2:nComb)) {
          locus <- indRecHist[[haplo]][comb, 2]
          rec_count[locus + ((chr - 1) * nLociAllPerChr), ] <- rec_count[locus + ((chr - 1) * nLociAllPerChr), ] + 1
        }
      }
    }
  }
}

rec_prob <- rec_count / (nInd * 2)

# ---- Write the files to disk ----

write.table(x = pedigree, file = "ped_file.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = genotypesObs_w_error, file = "geno_file.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(x = sequenceReads, file = "seq_file.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

write.table(x = haplotypes, file = "true-hap_0.5.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(x = phasedGenotypes, file = "true-phased_geno_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = UnphasedGenotypes, file = "true-geno_prob.txt",
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = genotypes, file = "true-geno_0.3333333333333333.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)
write.table(x = genotypes, file = "true-dosage.txt", 
            row.names = TRUE, col.names = FALSE, quote = FALSE)

write.table(x = geno_error, file = "true-geno_error_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = seq_error, file = "true-seq_error_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = alt_allele_prob, file = "true-alt_allele_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = segregation, "true-seg_prob.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(x = rec_prob, file = "true-rec_prob.txt",
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# ----- Map file -----

values <- data.frame(1, paste0(1, "-", 1:nLociAllPerChr), 1:nLociAllPerChr)
colnames(values) <- c("Chromosome number", "Marker name", "Base pair position")
for (chr in (2:nChr)) {
  if (chr < nChr) {
    value <- data.frame(chr, paste0(chr, "-", 1:nLociAllPerChr), (((chr - 1) * nLociAllPerChr + 1):(chr * nLociAllPerChr)))
  } else if (chr == nChr) {
    # needed when nLociAll is not divisible by nChr
    nLociAllLastChr <- (nLociAll - (nLociAllPerChr * (nChr - 1)))
    value <- data.frame(chr, paste0(chr, "-", 1:(nLociAllLastChr)), (((chr - 1) * nLociAllPerChr + 1):nLociAll))
  } else {
    # only one chromosome
    break
  }
  colnames(value) <- c("Chromosome number", "Marker name", "Base pair position")
  values <- rbind(values, value)
}
write.table(x = values, file = "map_file.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# ----- Segregation map file -----

subset <- floor(seq(1, nLociAll, length.out = nSegMap))
subsetValues <- values[subset,]
write.table(x = subsetValues, file = "seg_map_file.txt",
            row.names = FALSE, col.names = FALSE, quote = FALSE)

