library(data.table)
# args = commandArgs(trailingOnly = TRUE)
# prefix = args[1]

# prefix = "ldCoverage"
getFile = function(fileName) {
    mat = as.matrix(fread(fileName))
    mat = mat[order(mat[,1]),]
    return(mat)
}

true = getFile("data/true_genotypes.txt")
masked = getFile("data/genotypes.txt")

pedigree = getFile("data/pedigree.txt")

pedigree = cbind(pedigree,-1)
getGeneration = function(index) {
    if(pedigree[index,2] == 0) return(0)
    sire = pedigree[index, 2]
    return(pedigree[pedigree[,1] == sire, 4] + 1)
}

for(i in 1:nrow(pedigree)) {
    pedigree[i,4] = getGeneration(i)
}
generations = pedigree[,4]

ids = pedigree[,1]


stratifyByGeneration = function(true, masked, mat) {
    vals = lapply(unique(generations), function(gen) {
        subTrue = true[pedigree[,4] == gen,-1]
        subMasked = masked[pedigree[,4] == gen,-1]
        subMasked[subMasked == 9] = NA

        subMat = mat[pedigree[,4] == gen,-1]
        subMat[subMat == 9] = NA


        # Get accuracy/yield for initially missing loci.
        yield = mean(!is.na(subMat))
        yieldMissing = mean(is.na(subMasked) & !is.na(subMat))/mean(is.na(subMasked))
        acc = mean(subMat[is.na(subMasked)] == subTrue[is.na(subMasked)], na.rm=T)
        return(c(gen, yield, yieldMissing, acc))
    })

    vals = do.call("rbind", vals)
    colnames(vals) = c("generation", "yield", "yieldMissing", "acc")
    return(vals)
}

getSubset = function(mat, ids){
    mat = mat[mat[,1] %in% ids,]
    mat = mat[order(mat[,1]),]
    return(mat)
}


getAccuracy= function(fileName){
    mat = getFile(fileName)
    mat = getSubset(mat, ids)
    print(fileName)
    print(stratifyByGeneration(true, masked, mat))
}


getAccuracy("outputs/ai2.genotypes")
getAccuracy("outputs/pop_only.genotypes")
getAccuracy("outputs/ped_only.genotypes")

