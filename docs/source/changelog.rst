=========
Changelog
=========

[0.0.6] - 2026-08-??
====================

New features
------------

* Add support for X chromosome imputation with ``-x_chr``, including X chromosome genotype probabilities, haplotypes, and segregation probabilities
    (:pr:`66`, :user:`AprilYUZhang`, :user:`XingerTang`, :user:`gregorgorjanc`).

Bug fixes
---------

* Fix invalid indexing for population imputation
    (:pr:`66`, :user:`XingerTang`, :user:`gregorgorjanc`).

Maintenance
-----------

* Add X chromosome examples, documentation, functional tests, and accuracy tests
    (:pr:`66`, :user:`AprilYUZhang`, :user:`XingerTang`, :user:`gregorgorjanc`).


[0.0.5] - 2026-08-07
====================

Maintenance
-----------

* Restore the memory efficiency with
    * Optional segregation probabilities storage and
    * Reduced memory usage for genotype probabilities
    (:pr:`75`, :user:`XingerTang`, :user:`gregorgorjanc`).


[0.0.4] - 2026-06-12
====================

New features
------------

* Add ``-version`` parser
    (:pr:`65`, :user:`XingerTang`, :user:`gregorgorjanc`).

Bug fixes
---------

* Fix bug introduced in ``v0.0.3`` related to the genotype probabilities
    (:pr:`68`, :user:`XingerTang`).

Maintenance
-----------

* Add and improve sections of documentation
    * Introduction
    * Getting Started
    * Usage
    * Algorithm
    * Changelog
    (:pr:`63`, :user:`XingerTang`, :user:`gregorgorjanc`).

* Update boilerplate printout
    (:pr:`63`, :user:`XingerTang`, :user:`gregorgorjanc`).

[0.0.3] - 2025-12-04
====================

First release on `PyPI <https://pypi.org/project/AlphaImpute2>`_. Have all core functionalities for pedigree-based method as well as population-based method.
