==================
Algorithm Overview
==================

Here are some notes about how the peeling algorithm works.
For further details read:

.. [1] Whalen, A, Ros-Freixedes, R, Wilson, DL, Gorjanc, G, Hickey, JM. (2018). *Hybrid peeling for fast and accurate calling, phasing, and imputation with sequence data of any coverage in pedigrees*. Genetics Selection Evolution; doi: https://doi.org/10.1186/s12711-018-0438-2

Peeling basics
==============

The idea behind multi-locus iterative peeling is
to estimate an individual's genotype based on information from their own SNP gentoype data,
the genotypes of their parents, and the genotypes of their offspring.
This can be challenging because we have to combine these different sources of information and
there can be feedback loops that need to be managed when doing this.
For example, if a child's genotypes were imputed based on the genotypes of their parents,
we would not want to use those genotypes directly to re-impute the parent.
Instead we would want to use only the "new" information from the genotypes of the child
(and potentially their offspring).

To keep these sources of information separate,
we explicitly calculate and store three different values.
These are,

- anterior: genotype probabilities for the individual based on their parent's genotypes,
- penetrance: genotype probabilities for an individual based on their own genotypes, and
- posterior: genotype probabilities for an individual based on the offspring's genotypes.
  For the penetrance term you will often need the mate's genotypes as well.

For each of these values we store genotype probabilities over the four, phased genotype states,
that is, ``aa``, ``aA``, ``Aa``, and ``AA``.
In all cases the first allele is the allele inherited from the father,
and the second is the allele that was inherited from the mother.

One of the key things about inheritance and associated probability calculations
is that individual's inherit their chromosomes from their parents in large blocks.
If we knew which blocks an individual inherited,
we could use that information to help us determine what alleles the individual carried,
and also to determine which alleles their parents carried.
For each individual, we estimate these values using a segregation probabilities.

We code segregation probabilities in two forms,

- as the joint probability for both the paternal and maternal segregation
  (given by four values, ordered ``pp``, ``pm``, ``mp``, and ``mm``, where the ``p`` denotes the paternal allele,
  and the ``m`` denotes the maternal allele, while position is as before - the first allele inherited from father,
  meaning that ``pp`` denotes that individual inherited paternal allele from the father and mother),
- or as the probability for either the father or the mother
  (a pair of single values giving the probability that the individual inherits the maternal haplotype,
  that is, a seg=1 means that the individual inherits the maternal haplotype for that parent).

In a lot of places, we will rely heavily on calculating the "inheritance" or "transmission" probabilities.
These give the probability that a child inherits an allele based on the genotypes of their parents.
For some cases this is simple, for example, if the father is ``AA``, and the mother is ``aa``
then their offspring will be ``Aa`` since we know that the father will  transmit a copy of ``A`` and
the mother will transmit a copy of ``a`` so the resulting offspring will be ``a``.

Other cases may be more challenging and involve multiple possible outcomes
with associated probabilities.
If the father is heterozygous, ``aA``, and the segregation is unknown,
then there will be
a 50% probability that the offspring will inherit an ``a`` and
a 50% probability they inherit a ``A``.
If we take such a father and the mother is ``AA``
then the genotype probabilities of the offspring are:

::

    p(aa): 0.0
    p(aA): 0.5
    p(Aa): 0.0
    p(AA): 0.5

However we may know which haplotype the individual inherits.
If the segregation of the individual is ``mm``
(it inherited the maternal allele from both parents)
then the resulting genotype probabilities of an offspring
from an ``aA`` father and an ``AA`` mother are:

::

    p(aa): 0.0
    p(aA): 0.0
    p(Aa): 0.0
    p(AA): 1.0

Segregation probabilities are really helpful for determining which alleles an individual inherits
from their parents and are used for both the peel down (anterior) and peel up (posterior) steps.
The following sections outline how the penetrance, anterior, posterior, and segregation terms are calculated.

Penetrance
----------

The penetrance term gives the probability of each of the four phased genotypes
based on the direct information we have about an individual's genotype.
There are two ways to calculate this term:

1. Via a helper function ``tinyhouse.ProbMath.getGenotypeProbabilities_ind``, which is what ``AlphaPeel``, ``AlphaFamImpute``, ``AlphaAssign`` use, and takes into account sequence data.

2. Or via a somewhat hacky one line command, ``ind.peeling_view.setValueFromGenotypes(ind.peeling_view.penetrance)``. which is what currently is used.

The basic idea is that we want to take a genotype and turn it into genotype probabilities.

- If the individual is genotyped as a ``0``, that is ``aa``, we want to set ``p(aa)=1.0``.
- If the individual is genotyped ``1``, that is ``aA`` or ``Aa``, we want to set ``p(aA) = p(Aa) = 0.5``.
- If the individual is genotyped as a ``2``, that is ``AA``, we want to set ``p(AA) = 1.0``.

The difference between the heterozygous and homozygous states is that
with the heterozygous state there are two possible allele combinations that produce the same genotype.
In all of the cases we also want to add a small amount of noise to the estimate to allow for genotyping errors.
This error is often called as "penetrance" error.

Anterior
--------

The anterior term is calculated differently for the founders and non-founders.

The anterior term for founders in the population (individuals without any parents)
are set using the population minor allele frequency.
To set the term, we use:

::

    pedigree.setMaf()
    founder_anterior = ProbMath.getGenotypesFromMaf(pedigree.maf)
    founder_anterior = founder_anterior*(1-0.1) + 0.1/4
    for ind in pedigree:
        if ind.isFounder():
            ind.peeling_view.setAnterior(founder_anterior.copy())

For non-founders the anterior value is based on the genotypes of their parents.
The probability that the inherit an ``A`` allele from their father if their segregation is unknown is:

::

    P(a|father) = 0.5 p(aA) + 0.5 p(Aa) + 1 p(AA)

where ``p(aA)`` relates to the genotype probabilities of their father.
If the segregation value is known, this is instead:

::

    P(a|father) = seg p(aA) + (1-seg) p(Aa) + 1 p(AA)

We can generate genotype probabilities for the four phased genotype
as the product of the allele probabilities for each parent:

::

    p(aA) = p(a|father) p(A|mother)

Posterior
---------

The posterior term is a bit trickier than the anterior term.
The idea is that for each full-sub family (that is, each father-mother mate pair)
we will use their children to estimate their genotype  probabilities.
To do this, we construct the join probabilities of their parent's genotypes,
a 4x4 matrix of values.
We then calculate the posterior estimate for a single parent by
marginalizing the joint genotype probabilities by the genotypes of the other parents.

The joint genotypes are estimated by

.. math::
    p(g_{father}, g_{mother}|children) = \prod(p(g_{father}, g_{mother}|child)).

Simulation results using AlphaPeel have suggested that accuracy may be increased
by using the called genotype probabilities.
Because of this we call the individual's genotypes, haplotypes, and segregation values.
This has the added benefit of allowing us to use a look-up table to
produce the joint genotype probabilities at a given locus.

We then marginalize the genotypes for each parent by,

.. math::
    p(g_{father}|children) = \sum(p(g_{father}, g_{mother}|children)p(g_{mother})).

We assume that the posterior values for each family are independent.
This lets them calculate them separately for each family group, and
then sum them together to produce the final called genotype.
Because some individuals have a large number of offspring,
we calculate the joint genotypes on a log scale and then convert back in the marginalization step.

Segregation
-----------

We estimate the segregation values in a two step process. In the first step we create "point estimates" for the segregation values. In the second step we smooth the point estimates.

1. In step 1 we look to see if the child's haplotype for a single parent matches one of the parent's haplotypes, but not the other.

    - For example, if the parent is ``aA``, and the child is ``aa`` we will set the segregation value of ``pp`` and ``pm`` to ``1-e`` since that is consistent with the child inheriting the grand paternal allele (first allele) from their sire.

    - We also can consider the case where the child is unphased and heterozygote. In this case we see if a particular combination of parental haplotypes will produce a heterozygous offspring.

    - We used called genotypes in this step because an individual (and their parent's) genotypes are not statistically independent from each other at each loci. Using genotype probabilities (particularly for the parents) can produce erroneous results.

2. For the second step we use a standard forward-backward algorithm, lifted almost directly from AlphaPeel. The transmission rate determines how much uncertainty when moving from one loci to the next.

Specific code comments
======================

General math
------------

**Exponential + normalization:**
When working with the posterior values it is important to handle overflow/underflow.
We do this by treating most of the posterior values on a log scale.
To convert back from a log scale to a normal scale requires an exponential operation.
Most of the cases where we do this, we also need to normalize the resulting values so that they sum to one.
In order to prevent issues with underflows,
we first calculate the maximum value in the slice of the array that we are dealing with.
We then subtract the maximum value and take the exponential.
This means that the greatest value in the array will be set to :math:`exp(0)=1`.
Very small values may be set to zero, but this just indicates that they have vanishingly small probability.

Parallel
--------

In order to parallelize the code, we take this approach.
The overall idea is to find blocks of individuals
who can be updated at the same time (in parallel) and perform those updates.
In the context of peeling, we can break up individuals into discrete generations and
perform updates on all of the individuals in the same generation at the same time.
We use a ``ThreadPoolExecutor`` to perform tasks in parallel, and
use ``numba``'s ``nogil`` flag to get around python's global interpreter lock.

Because of the overhead in crating threads, and calling ``numba`` functions,
we split out the tasks in groups of full-sub families,
which will be updated at the same time.
Because a given parent can be a parent of multiple families
(but an offspring can only be an offspring of one family),
we set the genotypes for the parents separately in a non-parallel mode.
A complete breakdown of the main steps (and parallelization) is:

- **father and mother genotypes** (no parallel):
  We set the genotypes of the fathers and mothers in a non-parallel mode.
  Individual father and mothers may be used in multiple families within the same generation,
  and so setting them within a family block may lead to conflicts.
- **Child genotypes** (parallel):
  We set the genotypes of the children in parallel.
  This is safe because children are only members of a single full sib family
  (the family which contains their father and mother).
  We also collapse the posterior terms for individuals with offspring in this step
  when estimating the posterior term for their parents.
  This is done because all of the posterior terms for the individual
  should have been calculated by this point
  (the offspring's generation is always greater than their parent's).
- **Child segregation** (parallel):
  Individual child segregation values are re-estimated in parallel.
  These values only depend on the genotypes of the child and their parents.
  Their parents genotypes are fixed for the parallel blocks, and
  their genotypes are set on a family-by-family basis.
- **Child anterior term** (parallel):
  The anterior terms are estimated in parallel.
  These depend only on the parent genotype probabilities which fixed for a given generation.
- **Parent posterior** (parallel, but outside of ``numba``):
  We add the family posterior elements to each of the parents in parallel but outside of ``numba``.
  I think this should be safe because the code is run in multi-threaded mode,
  and so the GIL should make the append operations threadsafe.

Speed
-----

- Time seems to be split pretty evenly between the anterior and posterior computation terms.
- On the whole, the posterior term seems to dominate compared to the anterior term (usually by a factor of ~2).
- Estimating the segregation appears to be fairly low cost.
- There do not seem to be any obvious speed bottlenecks.

Memory
------

The storage of values takes place inside ``jit_Peeling_Individual`` in ``ImputationIndividual``. The main costs are:

- three ``4xnLoci float32`` s:
    - anterior
    - penetrance
    - genotypeProbabilities
- For individuals with offspring there are an additional two ``4xnloci float32`` s:
    - posterior
    - newPosterior
- one ``2xnLoci float32``:
    - segregation

There are a lot of possible places to obtain substantial memory savings.

- For all individuals
    - Because the anterior terms segregate independently, this could be reduced down to ``2xnLoci float32``,
      and re-calculated on the fly.
    - All of the ``4xnLoci float32`` s could potentially be stored as ``int8`` s with some conversion from ``int8->float32``.
      We don't actually need these terms to be very accurate (as long as we can still accurately call values).
- Individuals without offspring
    - For individuals without offspring we only ever use their called genotypes
      (with just the penetrance term) and
      segregation estimates in the peeling.
    - These terms come to play in the context of the posterior term for their parents.
    - This means we could potentially just save their genotypes, haplotypes, and segregation estimate.
    - If we need to call the individual in the future, we can re-calculate their anterior term on the fly, and
      recombine with the individual's genotype.
- For individuals with offspring:
    - We only ever use the penetrance+posterior or penetrance+anterior+posterior.
      We could calculate and store these values explicitly instead of storing them independently and
      re-calculating the genotype probabilities.
    - We currently store the posterior estimates as a list and re-add.
      We could instead store the values as a single matrix and just add each time.
      We need to be careful with the parallel updates on this term though.
