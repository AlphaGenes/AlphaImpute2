from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import Imputation_Programs

import datetime

try:
    profile
except:
    def profile(x): 
        return x

@profile
def main():
    
    ### Setup
    startTime = datetime.datetime.now()

    args = InputOutput.parseArgs("AlphaImpute")
    pedigree = Pedigree.Pedigree() 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True)
    
    # Fill in haplotypes from genotypes. Fill in genotypes from phase.
    Imputation_Programs.setupImputation(pedigree)

    print("Read in and initial setup", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    if not args.no_impute: 

        # Perform initial imputation + phasing before sending it to the phasing program to get phased.
        # The imputeBeforePhasing just runs a peel-down, with ancestors included.        
        print("Performing initial pedigree imputation")
        Imputation_Programs.imputeBeforePhasing(pedigree)
        print("Initial pedigree imputation finished.", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()
        
        # Phasing
        print("HD Phasing started.")
        if not args.no_phase:
            Imputation_Programs.phaseHD(pedigree)
        print("HD Phasing", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

        # Imputation.
        for genNum, gen in enumerate(pedigree.generations):
            # This runs a mix of pedigree based and population based imputation algorithms to impute each generation based on phased haplotypes.
            print("Generation:",  genNum)
            Imputation_Programs.imputeGeneration(gen, pedigree, args, genNum)
            print("Generation ", genNum, datetime.datetime.now() - startTime); startTime = datetime.datetime.now()


    #Write out.
    if args.binaryoutput :
        InputOutput.writeOutGenotypesPlink(pedigree, args.out)
    else:
        # pedigree.writeGenotypes(args.out + ".genotypes")
        pedigree.writeGenotypes_prefil(args.out + ".genotypes")
        pedigree.writePhase(args.out + ".phase")
    print("Writeout", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()


if __name__ == "__main__":
    main()