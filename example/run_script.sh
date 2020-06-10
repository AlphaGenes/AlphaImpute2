mkdir outputs
# AlphaImpute2 is a command line package for imputation in pedigree populations
# Install AlphaImpute2 via pip using:
# pip install <wheel name>

#To check command line arguments run AlphaImpute2 without any arguments.
AlphaImpute2

# Example 1: Run the full algorithm which runs both population and pedigree based imputation.

AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/ai2 \
             -maxthreads 4

# Example 1b: Run the population imputation algorithm only (used when pedigree data is unavailible).

AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/pop_only \
             -pop_only \
             -maxthreads 4

# Example 2: Run the pedigree based imputation algorithm only to get high-confidence phased haplotypes.

AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/ped_only \
             -ped_only \
             -final_peeling_threshold 0.98 \
             -phase_output \
             -maxthreads 4

