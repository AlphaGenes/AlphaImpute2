#!/bin/sh

# AlphaImpute2 is a command line package for imputation in pedigree populations
# Install AlphaImpute2 via pip using:
# pip install AlphaImpute2

# To see command line arguments run AlphaImpute2 without any arguments or providing -h or --help
# AlphaImpute2
# AlphaImpute2 -h
# AlphaImpute2 --help

# Below is a set of examples that give you a flavour on how to run and use AlphaImpute2
mkdir -p outputs

# Example 1: Run the full algorithm with population and pedigree based imputation
AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/ai2 \
             -maxthreads 4

# Example 1b: Run the population imputation algorithm only (used when pedigree data is unavailible)
AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/pop_only \
             -pop_only \
             -maxthreads 4

# Example 2: Run the pedigree based imputation algorithm and only call high-confident haplotypes
AlphaImpute2 -genotypes data/genotypes.txt \
             -pedigree data/pedigree.txt \
             -out outputs/ped_only \
             -ped_only \
             -final_peeling_threshold 0.98 \
             -phase_output \
             -maxthreads 4
