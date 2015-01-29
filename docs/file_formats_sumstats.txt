This document describes all new file formats introduced for use with the --h2 and --rg flags.

*NOTE* chromosomes are assumed to be integers. This software does not yet deal with sex
chromosomes.

(1) sumstats
---------
This file format is used for storing GWAS Z-scores. The format is Tab- delimited text, 
one row per SNP with a header row. The column order does not matter. We strongly 
recommend that you convert your summary statistics to the .sumstats format using the
munge_sumstats.py script included with ldsc, because munge_sumstats.py checks all the
gotchas that we've run into over the course of developing this software and applying it
to all of the data in Bulik-Sullivan*, Finucane*, et al, 2015.

Required Columns
(1) SNP -- SNP identifier (e.g., rs number)
(2) N -- sample size (which may vary from SNP to SNP).
(3) Z -- Z-score. Sign is w.r.t. A1
(4) A1 -- first allele
(5) A2 -- second allele

Note that LDSC filters out all variants that are not SNPs and strand-ambiguous SNPs. 