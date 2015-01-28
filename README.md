LDSC (LD SCore)
===============

v1.0.0

LDSC is a command line tool for estimating heritability and genetic correlation from GWAS
summary statistics. LDSC also computes LD Scores.

Support
-------

Before contacting us, please try the following:

1. Common issues are described in the [FAQ](docs/FAQ)
2. The methods are described in the papers (see citations below)
3. Search the [issue tracker](https://github.com/bulik/ldsc/issues)

Please report bugs on the [issue tracker](https://github.com/bulik/ldsc/issues). 

Citation
--------

If you use the software or the LD Score regression intercept, please cite

Bulik-Sullivan, B et al. LD Score Regression Distinguishes Confounding from Polygenicity in Genome-Wide Association Studies.
In Press at Nature Genetics. ([Director's cut](http://biorxiv.org/content/early/2014/02/21/002931))

For genetic correlation, please also cite

Bulik-Sullivan, et al. An Atlas of Genetic Correlations across Human Diseases and Traits. bioRxiv doi: http://dx.doi.org/10.1101/014498

For partitioned heritability, please also cite

Finucane, HK, et al. Partitioning Heritability by Functional Category using GWAS Summary Statistics. bioRxiv doi: http://dx.doi.org/10.1101/014241


Requirements
------------

1. Python 2.7
2. argparse 1.2.1
3. bitarray 0.8.1
4. numpy 1.8.0
5. pandas 0.15.0
6. scipy 0.10.1

License
-------

This project is licensed under GNU GPL v3.


Authors
-------

Brendan Bulik-Sullivan (Broad Institute)

Hilary Finucane (MIT Math)
