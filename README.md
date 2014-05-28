LDSC (LD SCore) v 1e-4
======================

Copyright (c) 2014 Brendan Bulik-Sullivan & Hilary Finucane

(Warning: under very active development)


What is LD Score?
--------------

LD Score is a measure of the amount of linkage disequilibrium (LD) around a SNP. 
LD Score allows one to properly take LD into account when attempting to make 
inferences about the genetic architecture of complex disease using GWAS summary 
statistics.

What can I do with LDSC?
---------------------

1. Estimate LD Score (and other, more exotic sums-of-moments of the genotype distribution).
2. Quantify the inflation in GWAS test statistics from confounding bias.
3. Estimate heritability from GWAS summary statistics.
4. Estimate partitioned heritability from GWAS summary statistcs.
5. Estimate genetic covariance and correlation from GWAS summary statistics.


Installation
------------

Prerequisites -- git, python 2.7, virtualenv.

1. Clone repository `git clone https://github.com/bulik/ldsc.git`
2. Create virtualenv `virtualenv ENV`
3. Activate virtualenv `source bin/activate`
4. Install requirements `pip install requirements.txt`
5. Now you should be able to call `python ldsc.py <flags>`
