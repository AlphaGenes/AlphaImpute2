.. AlphaImpute2 documentation master file, created by
   sphinx-quickstart on Thu Oct 10 10:16:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. NOTE:  added the line to the latex options:   'extraclassoptions': 'openany,oneside'

AlphaImpute2
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. highlight:: none

Introduction
~~~~~~~~~~~~


AlphaImpute2 is program to perform imputation in a range of animal and plant species. 

Please report any issues to `John.Hickey@roslin.ed.ac.uk <John.Hickey@roslin.ed.ac.uk>`_ or `awhalen@roslin.ed.ac.uk <awhalen@roslin.ed.ac.uk>`_.

Availability
------------

AlphaImpute2 is available from the `AlphaGenes <http://www.alphagenes.roslin.ed.ac.uk/software-packages/AlphaImpute2/>`_ website. The download files contains a python wheel file along with this documentation and an example. 

Conditions of use
-----------------

AlphaImpute2 is part of a suite of software that our group has developed. It is fully and freely available for all use under the MIT License.

Suggested Citation:

Whalen, A, Hickey, JM. (2020). *AlphaImpute2: Fast and accurate pedigree and population based imputation for hundreds of thousands of individuals in livestock populations*. BioRxiv https://doi.org/10.1101/2020.09.16.299677

Disclaimer
----------

While every effort has been made to ensure that AlphaImpute2 does what it claims to do, there is absolutely no guarantee that the results provided are correct. Use of AlphaImpute2 is entirely at your own risk.


Program Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

AlphaImpute2 takes in a number of command line arguments to control the program's behavior. To view a list of arguments, run AlphaImpute2 without any command line arguments, i.e. ``AlphaImpute2`` or ``AlphaImpute2 -h``. 

There are four primary ways to run AlphaImpute2 which differ on whether population or pedigree imputation should be run. The default option is to run both population and pedigree imputation in an integrated algorithm. This will be the option most users will want if they have access to pedigree data on a majority of individuals. The second option is to run population imputation only with the ``-pop_only`` flag. This option should be used if no pedigree data is availible. The third option is to run only pedigree based imputation using the ``-ped_only`` flag. This option is not recommended for general use cases, but may be applicable if (1) there are more than five generations of pedigree data, (2) imputation is done only on the most recent generations, (3) speed is a priority.

The fourth option is to run AlphaImpute2 with the ``-cluster_only``. This option performs AlphaImpute2's array clustering algorithm and outputs the results of the clustering. This option may be useful for debugging how individuals are clustered.


Core Arguments 
--------------

::
  
  Core arguments
    -out prefix              The output file prefix.

The ``-out`` argument gives the output file prefix for where the outputs of AlphaImpute2 should be stored. By default, AlphaImpute2 outputs a file with imputed genotypes, ``prefix.genotypes`` and phased haplotypes ``prefix.phase``. For more information on which files are created, see "Output Arguments", below.

Input Arguments 
----------------

::

    Input arguments:
      -bfile [BFILE ...]    A file in plink (binary) format. Only stable on Linux).
      -genotypes [GENOTYPES ...]
                            A file in AlphaGenes format.
      -reference [REFERENCE ...]
                            A haplotype reference panel in AlphaGenes format.
      -seqfile [SEQFILE ...]
                            A sequence data file.
      -pedigree [PEDIGREE ...]
                            A pedigree file in AlphaGenes format.
      -phasefile [PHASEFILE ...]
                            A phase file in AlphaGenes format.
      -startsnp STARTSNP    The first marker to consider. The first marker in the file is marker '1'. Default: 1.
      -stopsnp STOPSNP      The last marker to consider. Default: all markers considered.
      -seed SEED            A random seed to use for debugging.

AlphaImpute2 requires a genotype file and an optional pedigree file to run the analysis.

AlphaImpute2 supports binary plink files, ``-bfile``, genotype files in the AlphaGenesFormat, ``-genotypes``. A pedigree file may be supplied using the ``-pedigree`` option. 

Use the ``-startsnp`` and ``-stopsnp`` comands to run the analysis only on a subset of markers.

Binary plink files require the package ``alphaplinkpython``. This can be installed via ``pip`` but is only stable for Linux.

Imputation arguments: 
------------------------
::

    Impute options:
      -maxthreads MAXTHREADS
                            Number of threads to use. Default: 1.
      -binaryoutput         Flag to write out the genotypes as a binary plink
                            output.
      -phase_output         Flag to write out the phase information.
      -seg_output           Flag to write out the segmentation information.
      -pop_only             Flag to run the population based imputation algorithm
                            only.
      -ped_only             Flag to run the pedigree based imputation algorithm
                            only.
      -cluster_only         Flag to just cluster individuals into marker arrays
                            and write out results.
      -length LENGTH        Estimated map length for pedigree and population
                            imputation in Morgans. Default: 1 (100cM).

These options control how imputation is run. The ``-maxthreads`` argument can be used to allow multiple threads to be used for imputation. This argument can be set seperately from the ``-iothreads`` argument (above). The speed gains of using multiple threads is close to linear for population imputation, but is more limited for pedigree based imputation.

The ``-length`` argument controls the assumed length of the chromosome (in Morgans). We have found that imputation is largely insensitive to this value so keeping this value at its default of 1, should work in many cases. There are additional options to control the assumed recombination used for population based imputation (below).

The binary output option flags the program to write out files in plink binary format. Binary plink files require the package ``alphaplinkpython``. This can be installed via ``pip`` but is only stable for Linux. A fake map file is generated.

The remaining options control how AlphaImpute2 is run.

Pedigree imputation options 
-----------------------------
::

    Pedigree imputation options:
      -cycles CYCLES        Number of peeling cycles. Default: 4
      -final_peeling_threshold FINAL_PEELING_THRESHOLD
                            Genotype calling threshold for final round of peeling.
                            Default: 0.1 (best guess genotypes).


These options control how pedigree imputation is run for either the pedigree only algorithm, or the combined algorithm. ``-cycles`` controls the number of cycles of peeling that are perfromed. An additional very-high-confidence cycle is always performed in addition to the cycles specific here. We recommend using the default value of 4 cycles. Additional cycles seem to provide limited benifit in most pedigrees. 

The ``-final_peeling_threshold`` argument gives the genotype calling threshold for the final round of peeling. This applies to both the pedigree only or the combined algorithm. We recommend either using best guess genotypes (default with a cutoff of 0.1) or high confidence genotypes (with a cutoff of 0.95). Values that cannot be imputed with high enough confidence will be coded as missing.


Population imputation options 
---------------------------------
::

  Population imputation options:
    -n_phasing_cycles N_PHASING_CYCLES
                          Number of phasing cycles. Default: 5
    -n_phasing_particles N_PHASING_PARTICLES
                          Number of phasing particles. Defualt: 40.
    -n_imputation_particles N_IMPUTATION_PARTICLES
                          Number of imputation particles. Defualt: 100.
    -hd_threshold HD_THRESHOLD
                          Percentage of non-missing markers for an individual be
                          classified as high-density when building the haplotype
                          library. Default: 0.95.
    -min_chip MIN_CHIP    Minimum number of individuals on an inferred low-
                          density chip for it to be considered a low-density
                          chip. Default: 0.05
    -phasing_loci_inclusion_threshold PHASING_LOCI_INCLUSION_THRESHOLD
                          Percentage of non-missing markers per loci for it to
                          be included on a chip for imputation. Default: 0.9.
    -imputation_length_modifier IMPUTATION_LENGTH_MODIFIER
                          Increases the effective map length of the chip for
                          population imputation by this amount. Default: 1.
    -phasing_length_modifier PHASING_LENGTH_MODIFIER
                          Increases the effective map length of the chip for
                          Phasing imputation by this amount. Default: 5.
    -phasing_consensus_window_size PHASING_CONSENSUS_WINDOW_SIZE
                          Number of markers used to evaluate haplotypes when
                          creating a consensus haplotype. Default: 50.

These options control how population imputation is run. This algorithm uses a particle-based imputation approach where a number of particles are used to explore genotype combinations with high posterior probability. Increasing the number of particles can increase accuracy. USe the options, ``-n_phasing_particles`` and ``n_imputation_particles`` to change the number of particles run for phasing and imputation.

AlphaImpute2 uses a number of rounds of phasing in order to iteratively build a haplotype reference panel from the observed data. The argument ``-n_phasing_cycles`` controls the number of rounds that are used for phasing. In pilot testing we have found that the default value of 5 cycles tends to give good accuracy. Additional accuracy may be possible by slightly increasing this value.

To perfrom phasing and imputation, AlphaImpute2 selects high-density individuals to form the haplotype reference panel. ``-hd_threshold`` gives the percentage of non-missing markers the individual needs to carry to be included in the high-density reference panel.

Similar to ``-length`` the ``-imputation_length_modifier`` and ``-phasing_length_modifier`` control the assumed chromosome length for phasing and imputation. These values are applied multiplicatively to the ``-length`` option. We have found that imputation accuracy is not very sensitive to these values and recommend setting them to their default value.

When AlphaImpute2 is run, multiple particles are merged based on the particle's score in a window centered around each marker. ``phasing_consensus_window_size`` controls the size of the window. Increasing this value can increase imputation accuracy if the low-density panel is very sparse compared to the high-density panel.


Joint imputation options 
---------------------------------
::

  Joint imputation options:
    -chip_threshold CHIP_THRESHOLD
                          Proportion more high density markers parents need to
                          be used over population imputation. Default: 0.95
    -final_peeling_threshold_for_phasing FINAL_PEELING_THRESHOLD_FOR_PHASING
                          Genotype calling threshold for first round of peeling
                          before phasing. This value should be conservative..
                          Default: 0.9.

These options control how population and pedigree imputation are combined. As part of the combined algorithm, AlphaImpute2 detects a small number of "pseudo-founders" to impute using the population imputation algorithm. These "pseudo-founders" are selected by finding individuals with higher genotyping densities than their parents. AlphaImpute2 tries to be conservative in which individuals are selected as a "pseudo-founder" and the ``-chip_threshold`` parameter tells algorithm how many more non-missing markers the individuals needs compared to their parents to be considered a "pseudo-founder".

Similar to the ``-final_peeling_threshold`` argument, the ``-final_peeling_threshold_for_phasing`` argument gives the final peeling threshold for the initial round of pedigree imputation in the combined algorithm. 

Input file formats
~~~~~~~~~~~~~~~~~~

Genotype file 
-------------

Genotype files contain the input genotypes for each individual. The first value in each line is the individual's id. The remaining values are the genotypes of the individual at each locus, either 0, 1, or 2 (or 9 if missing). The following examples gives the genotypes for four individuals genotyped on four markers each.

Example: ::

  id1 0 2 9 0 
  id2 1 1 1 1 
  id3 2 0 2 0 
  id4 0 2 1 0

Pedigree file
-------------

Each line of a pedigree file has three values, the individual's id, their father's id, and their mother's id. "0" represents an unknown id.

Example: ::

  id1 0 0
  id2 0 0
  id3 id1 id2
  id4 id1 id2

Phase file
-----------

The phase file gives the phased haplotypes (either 0 or 1) for each individual in two lines. For individuals where we can determine the haplotype of origin, the first line will provide information on the paternal haplotype, and the second line will provide information on the maternal haplotype.

Example: ::

  id1 0 1 9 0 # Maternal haplotype
  id1 0 1 9 0 # Paternal haplotype
  id2 1 1 1 0
  id2 0 0 0 1
  id3 1 0 1 0
  id3 1 0 1 0 
  id4 0 1 0 0
  id4 0 1 1 0

  
Binary plink file
-----------------

AlphaImpute2 supports the use of binary plink files using the package ``AlphaPlinkPython``. AlphaImpute2 will use the pedigree supplied by the ``.fam`` file if a pedigree file is not supplied. Otherwise the pedigree file will be used and the ``.fam`` file will be ignored. 


Output file formats
~~~~~~~~~~~~~~~~~~~

Genotype file 
-------------

Genotype files contain the input genotypes for each individual. The first value in each line is the individual's id. The remaining values are the genotypes of the individual at each locus, either 0, 1, or 2 (or 9 if missing). The following examples gives the genotypes for four individuals genotyped on four markers each.

Example: ::

  id1 0 2 9 0 
  id2 1 1 1 1 
  id3 2 0 2 0 
  id4 0 2 1 0

Phase file
-----------

The phase file gives the phased haplotypes (either 0 or 1) for each individual in two lines. For individuals where we can determine the haplotype of origin, the first line will provide information on the paternal haplotype, and the second line will provide information on the maternal haplotype.

Example: ::

  id1 0 1 9 0 # Maternal haplotype
  id1 0 1 9 0 # Paternal haplotype
  id2 1 1 1 0
  id2 0 0 0 1
  id3 1 0 1 0
  id3 1 0 1 0 
  id4 0 1 0 0
  id4 0 1 1 0
