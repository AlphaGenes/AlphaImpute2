from .tinyhouse import Pedigree
from .tinyhouse import InputOutput

from .Imputation import Imputation_Programs
from .Imputation import Imputation
import numpy as np
import random
import datetime

# from Util import HapLibParams, createHapLibParams
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
    Imputation_Programs.setupImputation(pedigree)

    print("Read in and initial setup", datetime.datetime.now() - startTime); startTime = datetime.datetime.now()

    if not args.no_impute: 
        
        #Do some small imputation before phasing. Phase. Then impute.
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