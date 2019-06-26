Here are some notes about how the hueristic peeling algorithm works. For more technical details it may be worth reading:

Hybrid peeling for fast and accurate calling, phasing and imputation with sequence data of any coverage in pedigrees, Whalen, A., R. Ros-Freixedes, D. L. Wilson, G. Gorjanc, J.M. Hickey, Genetics Selection Evolution

Peeling basics
=====

The idea behind multi-locus iterative peeling is to estimate an individual's genotype based on information from their own SNP data, the genotypes of their parents, and the genotypes of their offspring. This can be hard because there are feedback loops that need to be avoided when doing this. For example, if a child's genotypes were imputed based on the genotypes of their parents, we would not want to use those genotypes to re-impute the parent. Instead we would want to use only the "new" data from the genotype's of the child (and potentially their offspring).

To keep these quantities seperate, we explicitly calculate and store three different values. These are,

- anterior: genotype probabilities for the individual based on their parent's genotypes.
- penetrance: genotype probabilities for an individual based on their own genotypes.
- posterior: genotype probabilites for an individual based on the offspring's genotypes. For the penetrance term you will often need the mate's genotypes as well.

For each of these values we store genotype probabilities over the four, phased genotype states, i.e. aa, aA, Aa, AA. In all cases the first allele is the allele inherited from the sire, and the second is the allele that was inherited from the dam.

One of the key things about imputation is that individual's inherit their chromosomes from their parents in large blocks. If we knew which blocks an individual inherited, we could use that information to help us determine what alleles the individual carried, and also to determine which alleles their parents carried. For each individual, we estimate these values using a "segregation probability". We code segregation probabilities in two forms, either as the joint probability for both the paternal and maternal segregation (given by four values, ordered  `pp, pm, mp, mm` where the first value is for the sire, and an "m" stands for the parent's maternal haplotype), or as the probability for either the sire or the dam (a pair of single values giving the probability that the individual inherits the maternal haplotype, i.e. a seg=1 means that the individual inherits the maternal haplotype for that parent).

In a lot of places, we will rely pretty heavily on calculating "inheritance" probabilities. These give the probability that a child inherits an allele based on the genotypes of their parents. For some cases this is simple, e.g., if the sire is  `AA`, and the dam is `aa` then the child will be `Aa`; we know that both parents will transmit a single allele and so the sire will  transmit a copy of `A` and the dam will transmit a copy of `a` so the resulting offspring will be  `Aa`.

Other cases may be more stochastic. If the sire is heterozygous, `aA`, and the segregation is unknown, then there will be a 50% probability that the child will inherit an `a` and a 50% probability they inherit a `A`. If the sire is `aA` and the dam is `AA` then the genotype probabilities of the offspring will be:
```
p(aa): 0
p(aA): .5
p(Aa): 0
p(AA): .5
```
However we may know which haplotype the individual inherits. If the segregation of the individual is `mm` (i.e. inherits the maternal haplotype from both parents ) then the resulting genotype probabilities for an  `aA` sire and a  `AA` dam are:
```
p(aa): 0
p(aA): 0
p(Aa): 0
p(AA): 1
```
Segregation probabilities are really helpfull for determining which alleles an individual inherits from their parents and are used for both the peel down (anterior) and peel up (posterior) steps. The following sections outline how the Penetrance, Anterior, Posterior, and Segregation terms are calculated.

Penetrance
--

The penetrance term gives the probability of each of the four phased genotypes based on the direct information we have about an individual's genotype. There are two ways to calculate this term:

1. Via a helper function  `tinyhouse.ProbMath.getGenotypeProbabilities_ind`, which is what AlphaPeel, AlphaFamImpute, AlphaAssign use, and takes into account sequene data.
2. Or via a somewhat hacky one line command,  `ind.peeling_view.setValueFromGenotypes(ind.peeling_view.penetrance)`. which is what currently is used.

The basic idea is that we want to take a genotype and turn it into genotype probabilities. If the individual is genotyped as a "0" we want to set `p(aa)=1`. If the individual is genotyped"1", we want `p(aA) = p(Aa) = .5`. If the individual is genotyped as a "2" we want to set `p(AA) = 1`. The difference between the heterozygous and homozygous states is that with the heterozygous state there are two possible phasings that will produce the same genotype. In all of the cases we also want to add a small amount (roughly 1%) of noise to the estimate to allow for genotyping errors.

Anterior
--

The anterior term is calculated differently for the founders and non founders.

The anterior term for founders in the population (individuals without any parents) are set using the population minor allele frequency. To set the term, we use:

```
pedigree.setMaf()
founder_anterior = ProbMath.getGenotypesFromMaf(pedigree.maf)
founder_anterior = founder_anterior*(1-0.1) + 0.1/4
for ind in pedigree:
    if ind.isFounder():
        ind.peeling_view.setAnterior(founder_anterior.copy())
```

For non-founders the anterior value is based on the genotypes of their parents. The probability that the inherit an A allele from their sire if their segregation is unknown is

```
P(a|sire) = .5 p(aA) + .5 p(Aa) + 1 p(AA),
```
where `p(aA)` relates to the genotype probabilities of their sire. If the segregation value is known, this is instead:
```
P(a|sire) = seg p(aA) + (1-seg) p(Aa) + 1 p(AA).
```
We can generate genotype probabilities for the four phased genotype as the product of the allele probabilities for each parent:
```
p(aA) = p(a|Sire) p(A|dam).
```


Posterior
--

The posterior term is a bit trickeir than the anterior term. The idea is that for each full-sub family (i.e. each sire-dam mate pair) we will use their children to estimate their genotype  probabilities. To do this, we construct the join probabilities of their parent's genotypes, a 4x4 matrix of values. We then calculate the posterior estimate for a single parent by marginalizing the joint genotype probabilities by the genotypes of the other parents.

The joint genotypes are estimated by
```
p(g_sire, g_dam|children) = \prod(p(g_sire, g_dam|child)).
```
Simulation results using AlphaPeel have suggested that accuracy may be increased by using the called genotype probabilities. Because of this we call the indivdiual's genotypes, haplotypes, and segregation values. This has the added benifit of allowing us to use a look-up table to produce the joint genotype probabilities at a given locus.

We then marginalize the genotypes for each parent by,
```
p(g_sire|children) = \sum(p(g_sire, g_dam|children)p(g_dam)).
```

We assume that the posterior values for each family are independent. This lets them calculate them seperately for each family group, and then sum them together to produce the final called genotype. Because some individuals have a large number of offspring, we calculate the joint genotypes on a log scale and then convert back in the marginalization step.

Segregation
--
We estimate the segregation values in a two step process. In the first step we create "point estimates" for the segregation values. In the second step we smooth the point estimates.

1. In step 1 we look to see if the child's haplotype for a single parent matches one of the parent's haplotypes, but not the other.
    * For example, if the parent is `aA`, and the child is `aa` we will set the segregation value of `pp` and `pm` to `1-e` since that is consistent with the child inheriting the grand paternal allele (first allele) from their sire.
    * We also can consider the case where the child is unphased and heterozygote. In this case we see if a particular combination of parental haplotypes will produce a heterozygous offspring.
    * We used called genotypes in this step because an individual (and their parent's) genotypes are not statistically independent from each other at each loci. Using genotype probabilities (particularly for the parents) can produce erroneous results.
2. For the second step we use a standard forward-backward algorithm, lifted almost directly from AlphaPeel. The transmission rate determines how much uncertainty when moving from one loci to the next.
    * Add some math here?


Specific code comments
==

General Math
--

**Exponential + normalization:** When working with the posterior values it is important to handle overflow/underflow appropriately. We do this by treating most of the posterior values as if they exist on a log scale. To convert back from a log scale to a normal scale requires an exponential operation. Most of the cases where we do this, we also need to normalize the resulting values so that they sum to one. In order to prevent issues with underflows, we first calculate the maximum value in the slice of the array that we are dealing with. We then subtract the maximum value and take the expontential. This means that the greatest value in the array will be set to `exp(0)=1`. Very small values may be set to zero, but this just indicates that they have vanishingly small probability.

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

- anterior
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
- Could just store the anterior values as 2xnLoci (i.e. for each haplotype seperately).
- We only ever set values to either `setGenotypesAll`, or `setGenotypesPosterior`. Potentially we could just store values for both of those terms, and have genotypeProbabilities point to one or two of those values. For the parents we'd also likely need the newPosterior term. ~2x space saving.
- I want to wait until we get a clearer picture of how the other algorithms work before moving forward on any of these ideas.
- We can also remove the memory costs when switching over to a different part of the algorithm by removing the `jit_Peeling_Individual`s.








