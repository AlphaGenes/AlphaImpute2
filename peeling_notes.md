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
```
p(aa): 0
p(aA): .5
p(Aa): 0
p(AA): .5
```
However we may know which haplotype the individual inherits. If the segregation of the individual is `mm` (i.e. inherits the maternal haplotype from both parents ) then the resulting genotype probabilities are:
```
p(aa): 0
p(aA): 0
p(Aa): 0
p(AA): 1
```
Segregation probabilities are really helpfull for determining which alleles an individual inherits from their parents and are used for both the peel down (anterior) and peel up (posterior) steps.

Penetrance, Anterior, and Posterior
==

This section will outline how each of these terms is calculated.

Penetrance
--

This gives the genotype probability of an individual given their own genotype. There are two ways to calculate this term.

1. There is a helper function, `tinyhouse.ProbMath.getGenotypeProbabilities_ind`, that is probably what should be used. This has the advantage of being able to take advantage of sequence data.
2. What is currently done is we use `ind.peeling_view.setValueFromGenotypes(ind.peeling_view.penetrance)`. This second function takes an individual's current genotype and haplotypes at sets their genotype probabiliites from those values.

Anterior
--

The anterior term is calculated differently for the founders and non founders.

For the founders we use the minor allele frequency in the population to set the anterior terms. To do this we use

```
pedigree.setMaf()
founder_anterior = ProbMath.getGenotypesFromMaf(pedigree.maf)
founder_anterior = founder_anterior*(1-0.1) + 0.1/4
for ind in pedigree:
    if ind.isFounder():
        ind.peeling_view.setAnterior(founder_anterior.copy())
```

For non-founders we calculate the anterior value based on the genotypes of their parents. To do this we calculate the probability that the inherit an A allele from their sire (based on the genotype probabilities for their sire). If their segregation value is missing this is,

```
P(a|sire) = .5 p(aA) + .5 p(Aa) + 1 p(AA).
```
If their segregation value is known, this turns into
```
P(a|sire) = seg p(aA) + (1-seg) p(Aa) + 1 p(AA).
```
We then can get genotype probabilities via, e.g.:
```
p(aA) = p(a|Sire) p(A|dam)
```


Posterior
--

The posterior terms are a bit tricky. The basic idea is that for each full-sib family (i.e. each sire-dam mate pair) we use the children to estimate their joint genotype probabilities. We use the called genotypes (and segregation) values to do this. This means that at each loci, each child will produce a 4x4 matrix of values that represent the joint probabilities for the phased genotypes of their sire and dam. We assume that alleles are inherited independently, i.e.
```
p(sire, dam|children) = \prod(p(sire, dam|child))
```
We then collapse the scores for each parent using the genotype probabiliites of the other parent, i.e.
```
p(sire|children) = \sum(p(sire, dam|children)p(dam)).
```
Because each family assorts independently we calculate the posterior terms seperately for each parent, and then collapse the values together to get the value for the parent.


Segregation
==

Estimating the segregation values is a two step process. First we determine which haplotype the individual inherited from at each loci, then we smooth the estimates to get the actual haplotype inheritance.

- For the first step we look at the child's genotypes/haplotypes. In each case we determine which of the segregation will produce the child's genotype and assign an `1-e` term to those that do and an `1-e` term to those that do not. We ignore loci where the child's haplotype is missing, or where one of the parent's haplotypes is missing. Generally we rely on haplotypes, however if the child is heterozygous, this can still provide some information particularly if one of the parents is phased and heterozygous.
- We used called genotypes in this step. This is because an individual's genotypes are not statistically independent from each other. This can mean that genotype probabilities at each loci can give eroneous results since their probabilities are not independent.
- For the second step we use a standard forward-backward algorithm, lifted almost directly from AlphaPeel. The transmission rate determines how much uncertainty is added at each transmission step.

Some general code comments
==

General Math
--

**Exponential + normalization:** This occurs when collapsing the posterior values together. When taking the exponential of very small quantities it is important to make sure that you do not run into overflow/underflow issues. In order to get around this, we can subtract the maximum value of the array from each of the elements, and then take the exponential of the resulting matrix. This means that the greatest value will get set to `exp(0)=1`. Very small values will get set to zero, but this indactes that they should have close to zero weight anyways, so it's probably okay.

Parallel
--

In order to parallelize the code we've split out each of the anterior and posterior steps into calculating the terms for individual families. We split out the families into individual generations. The key thing when doing this is that we don't want to be setting values for the same individual in mulitple threads. Here is an analysis split out between different values.

- **Sire and dam genotypes** (no parallel): We set the genotypes of the sires and dams in a non-parallele mode. Individual sire and dams may be used in multiple families within the same generation.
- **Child genotypes** (parallel): We set the genotypes of the children in parallel. This is safe because we are operating on a family basis, and a child is only the offspring of a single family (i.e. the family that has their sire as sire, and dam as dam). We also collapse the posterior term for the individual in this step.-
- **Child segregation** (parallel): Individual child segregation values are re-estimated in parallel. These depend only on the genotypes of the child and parents which are fixed.
- **Child anterior term** (parallel): The anterior terms are estimate in parallel. These depend only on the parent genotype probabilties which are fixed in the context of a generation.
-**Parent posterior**(parallel, but outside of numba): We add the family posterior elements to each of the parents in parallel but outside of numba. I think this should be safe because the code is run in multi-threaded mode, and so the GIL should make these operations threadsafe.

Speed
--

- Time seems to be split pretty evenly between the anterior and posterior computation terms.
- On the whole, the posterior term seems to dominate compared to the anterior term (usually by a factor of ~2).
- Estimating the segregation seems to be fairly cheap.
- There do not seem to be any obvious speed gains.

Memory
--

The memory storage takes place inside `jit_Peeling_Individual` in `ImputationIndividual`. The main costs are:

3 4xnLoci float32s:
-anterior
- penetrance
- genotypeProbabilities

If the individual is a parent there are an additional 2 4xnloci float32s:
- posterior
- newPosterior

1 2xnLoci float32
- segregation

Potentially there are some gains to be made here.
- all of the 4xnLoci float32s could potentially be stored as int8s with some conversion from int8->float32. We don't actually need these terms to be very accurate (as long as we can still accurately call values). 4x space saving.
- Potentially we could drop the penetrance term and re-calculate from original genotypes/haplotypes. My main concern is that I've thought about using the pentrance term as a way to store values from outside algorithms. I'm somewhat relctant to make that change before we have integration done.
- We only ever set values to either `setGenotypesAll`, or `setGenotypesPosterior`. Potentially we could just store values for both of those terms, and have genotypeProbabilities point to one or two of those values. For the parents we'd also likely need the newPosterior term. ~2x space saving.
- I want to wait until we get a clearer picture of how the other algorithms work before moving forward on any of these ideas.









