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

AlphaImpute2 is part of a suite of software that our group has developed. It is fully and freely available for academic use, provided that users cite it in publications. However, due to our contractual obligations with some of the funders of the research program that has developed this suite of software, all interested commercial organizations are requested to contact John Hickey (`John.Hickey@roslin.ed.ac.uk <John.Hickey@roslin.ed.ac.uk>`_) to discuss the terms of use.

Suggested Citation:

TO BE ADDED LATER

Disclaimer
----------

While every effort has been made to ensure that AlphaImpute2 does what it claims to do, there is absolutely no guarantee that the results provided are correct. Use of AlphaImpute2 is entirely at your own risk.


Program Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

AlphaImpute2 takes in a number of command line arguments to control the program's behavior. To view a list of arguments, run AlphaImpute2 without any command line arguments, i.e. ``AlphaImpute2`` or ``AlphaImpute2 -h``. 


Core Arguments 
--------------

::
  
  Core arguments
    -out prefix              The output file prefix.

The ``-out`` argument gives the output file prefix for where the outputs of AlphaImpute2 should be stored. By default, AlphaImpute2 outputs a file with imputed genotypes, ``prefix.genotypes``, phased haplotypes ``prefix.phase``, and genotype dosages ``prefix.dosages``. For more information on which files are created, see "Output Arguments", below.

Input Arguments 
----------------

::

    Input arguments:
      -bfile [BFILE [BFILE ...]]
                            A file in plink (binary) format. Only stable on
                            Linux).
      -genotypes [GENOTYPES [GENOTYPES ...]]
                            A file in AlphaGenes format.
      -reference [REFERENCE [REFERENCE ...]]
                            A haplotype reference panel in AlphaGenes format.
      -seqfile [SEQFILE [SEQFILE ...]]
                            A sequence data file.
      -pedigree [PEDIGREE [PEDIGREE ...]]
                            A pedigree file in AlphaGenes format.
      -phasefile [PHASEFILE [PHASEFILE ...]]
                            A phase file in AlphaGenes format.
      -startsnp STARTSNP    The first marker to consider. The first marker in the
                            file is marker "1".
      -stopsnp STOPSNP      The last marker to consider.
      -seed SEED            A random seed to use for debugging.

AlphaImpute2 requires a genotype file and an optional pedigree file to run the analysis.

AlphaImpute2 supports binary plink files, ``-bfile``, genotype files in the AlphaGenesFormat, ``-genotypes``. A pedigree file may be supplied using the ``-pedigree`` option. 

Use the ``-startsnp`` and ``-stopsnp`` comands to run the analysis only on a subset of markers.

Binary plink files require the package ``alphaplinkpython``. This can be installed via ``pip`` but is only stable for Linux.

Output Arguments 
----------------
::

    Output options:
      -writekey WRITEKEY    Determines the order in which individuals are ordered
                            in the output file based on their order in the
                            corresponding input file. Animals not in the input
                            file are placed at the end of the file and sorted in
                            alphanumeric order. These animals can be suppressed
                            with the "-onlykeyed" option. Options: id, pedigree,
                            genotypes, sequence, segregation. Defualt: id.
      -onlykeyed            Flag to suppress the animals who are not present in
                            the file used with -outputkey. Also suppresses "dummy"
                            animals.
      -iothreads IOTHREADS  Number of threads to use for io. Default: 1.


The order in which individuals are output can be changed by using the ``writekey`` option. This option changes the order in which individuals are written out to the order in which they were observed in the corresponding file. The ```-onlykeyed`` option suppresses the output of dummy individuals.

The parameter ``-iothreads`` controls the number of threads/processes used by AlphaImpute2. AlphaImpute2 uses additional threads to parse and format input and output files. Setting this option to a value greater than 1 is only recommended for very large files (i.e. >10,000 individuals).


Imputation arguments: 
------------------------
::

    Impute options:
      -maxthreads MAXTHREADS
                            Number of threads to use. Default: 1.
      -binaryoutput         Flag to write out the genotypes as a binary plink
                            output.
      -phase_output         Flag to write out the phase information.
      -pop_only             Flag to run the population based imputation algorithm
                            only.
      -ped_only             Flag to run the pedigree based imputation algorithm
                            only.
      -cluster_only         Flag to just cluster individuals into marker arrays
                            and write out results.

These give various options to run imputation.


Pedigree imputation options 
-----------------------------
::

    Pedigree imputation options:
      -cycles CYCLES        Number of peeling cycles. Default: 4
      -final_peeling_threshold FINAL_PEELING_THRESHOLD
                            Genotype calling threshold for final round of peeling.
                            Default: 0.1 (best guess genotypes).


Options for pedigree based imputation



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
                            Percentage of non-missing markers to be classified as
                            high-density when building the haplotype library.
                            Default: 0.95.
      -min_chip MIN_CHIP    Minimum number of individuals on an inferred low-
                            density chip for it to be considered a low-density
                            chip. Default: 0.05


Options for population based imputation


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
      -integrated_decision_rule INTEGRATED_DECISION_RULE
                            Decision rule to use when determining whether to use
                            population or pedigree imputation. Options:
                            individual, balanced, parents. Default: individual
      -joint_type JOINT_TYPE
                            Decision rule to use when determining which joint
                            option to use. Options: integrated, pedigree. Default:
                            pedigree


Options for running both pedigree and population imputation together.


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

Binary plink file
-----------------

AlphaImpute2 supports the use of binary plink files using the package ``AlphaPlinkPython``. AlphaImpute2 will use the pedigree supplied by the ``.fam`` file if a pedigree file is not supplied. Otherwise the pedigree file will be used and the ``.fam`` file will be ignored. 


Map file 
-----------------

The map file gives the chromosome number and the marker name and the base pair position for each marker in two columns. AlphaImpute2 needs to be run with all of the markers on the same chromosome. 

Example: ::

  1 snp_a 12483939
  1 snp_b 192152913
  1 snp_c 65429279
  1 snp_d 107421759


Output file formats
~~~~~~~~~~~~~~~~~~~

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
