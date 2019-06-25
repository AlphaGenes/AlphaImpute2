Here are some notes about how the hueristic peeling algorithm works. For more technical details it may be worth reading:

Hybrid peeling for fast and accurate calling, phasing and imputation with sequence data of any coverage in pedigrees, Whalen, A., R. Ros-Freixedes, D. L. Wilson, G. Gorjanc, J.M. Hickey, Genetics Selection Evolution

Peeling basics
=====

Fundementally the idea behind multi-locus peeling is to calculate an individual's genotype as a product of the genotypes of their parents and offspring. The tricky thing is to prevent feedback loops while doing this. I.e. imagine an child was imputed based on the genotypes of the parents, we do not want to use the imputed genotypes of the child to be used when imputing their parents. This will cause us to be overly certain about the genotypes of the parents and may lead to errors.

For all of these peeling quantities, we want to generate genotype probabilities over the four, phased genotypes, i.e.
genotype probabilities: [p(aa), p(aA), p(Aa), p(AA)].
In all cases the first allele is the allele the individual inherited from their sire, and the second is inherited from their dam.

There are three quantities that we calculate.

anterior: genotype probabilities for the individual based on their parent's genotypes.
penetrance: genotype probabilities for an individual based on their own genotypes.
posterior: genotype probabilites for an individual based on the offspring's genotypes. For the penetrance term you will often need the mate's genotypes as well.

In order to help us calculate these quantities we will also calculate a "segregation probability". This value gives the probility that the child inherited which haplotype from which parent at which loci. In our code the ordering is given by:
```pp, pm, mp, mm```
Note: These orderings are helpfully given in Heuristic_Peeling.def smoothPointSeg.

Where the first value is for the sire, the second for the dam. An "m" stands for the grandmaternal haplotype, and a "p" stands for the grand paternal haplotype. There will be a more explicit example in a few paragraphs.

In order to calculate a lot of these quantities we'll use information about how loci are transmitted between an individual and their offspring. Some biology will be useful here. At each loci, an individual will generally have two alleles. These alleles will be on one of two copies of an individual chromosomes. We will call these copies "haplotypes". What we care about is given a parent's haplotype, what is the probability of the offspring's haplotype?

For some transmission cases it's pretty simple, i.e. imagine the sire is `AA`, and the dam is `aa`. We know that both parents transmit at least one copy of each allele. These means that the sire will transmit a copy of `A` and the dam will transmit a copy of `a`. The resulting offspring will be `Aa`, where the `A` comes from the sire, and the `a` comes from the dam.

For other cases it may be more complicated. If the sire is `aA`, and we do not know which haplotype the child inherits at which loci, then there is a 50% chance that the child will inherit an `a` and a 50% chance they inherit a `A`. If the sire is `aA` and the dam is `AA` then the genotype probabilities of the offspring will be:
p(aa): 0
p(aA): .5
p(Aa): 0
p(AA): .5

However we may know which haplotype the individual inherits. If the segregation of the individual is `mm` (i.e. inherits the maternal haplotype from both parents ) then the resulting genotype probabilities are:

p(aa): 0
p(aA): 0
p(Aa): 0
p(AA): 1

Segregation probabilities are really helpfull for determining which alleles an individual inherits from their parents and are used for both the peel down (anterior) and peel up (posterior) steps.

Penetrance, Anterior, and Posterior
==

This section wi

