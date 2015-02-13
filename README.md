
#LDSC (LD SCore) `v1.0.0`

`ldsc` is a command line tool for estimating heritability and genetic correlation from GWAS summary statistics. `ldsc` also computes LD Scores.

## Getting Started

First, you will need to install python as well as the packages listed under the requirements header below. The easiest way to do this is with the [Anaconda](https://store.continuum.io/cshop/anaconda/) python distribution. All of the required packages come standard with Ananconda.

In order to download `ldsc`, you should clone this repository via the command
'''
  git clone https://github.com/bulik/ldsc.git
'''
Once you have installed `ldsc` as well as the required packages, typing
```
$ python ldsc.py -h
```
will print a list of all command-line options. Short tutorials describing the four basic functions of `ldsc` (estimating LD Scores, h2 and partitioned h2, genetic correlation, the LD Score regression intercept) can be found in the wiki. If you would like to run the tests, please see the wiki.

## Where Can I Get LD Scores?

You can download LD Scores that are suitable for basic LD Score analyses (the LD Score regression intercept, heritability, genetic correlation) [here](http://www.broadinstitute.org/~bulik/eur_ldscores/).


##Support

Before contacting us, please try the following:

1. The [tutorials](tutorials/) have basic advice on running `ldsc` and interpreting the output
2. Common issues are described in the [FAQ](docs/FAQ)
2. The methods are described in the papers (citations below)

If that doesn't work, you can get in touch with us via the [google group](https://groups.google.com/forum/?hl=en#!forum/ldsc_users).


##Citation

If you use the software or the LD Score regression intercept, please cite

[Bulik-Sullivan, et al. LD Score Regression Distinguishes Confounding from Polygenicity in Genome-Wide Association Studies.
Nature Genetics, 2015.](http://www.nature.com/ng/journal/vaop/ncurrent/full/ng.3211.html)

For genetic correlation, please also cite

Bulik-Sullivan, et al. An Atlas of Genetic Correlations across Human Diseases and Traits. bioRxiv doi: http://dx.doi.org/10.1101/014498

For partitioned heritability, please also cite

Finucane, HK, et al. Partitioning Heritability by Functional Category using GWAS Summary Statistics. bioRxiv doi: http://dx.doi.org/10.1101/014241


##Requirements

1. `Python 2.7`
2. `argparse 1.2.1`
3. `bitarray 0.8.1`
4. `numpy 1.8.0`
5. `pandas 0.15.0`
6. `scipy 0.10.1`

##License

This project is licensed under GNU GPL v3.


##Authors

Brendan Bulik-Sullivan (Broad Institute of MIT and Harvard)

Hilary Finucane (MIT Department of Mathematics)
