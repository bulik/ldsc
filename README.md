LDSC (LD SCore) v 0.0.1 (alpha)
==============================

Copyright (c) 2014 Brendan Bulik-Sullivan & Hilary Finucane

(NB We're still working on it. The dev branch is going to become v1.0.0 any day now. To report bugs, please either raise an issue on github or email us with LDSC in the subject line and the log file copied into the body of your message. Note that the in-progress tutorials on the wiki refer to the dev branch).

What can I do with LDSC?
---------------------

1. Estimate LD Score
2. Quantify the inflation in GWAS test statistics from confounding bias.
3. Estimate heritability from GWAS summary statistics.
4. Estimate partitioned heritability from GWAS summary statistcs.
5. Estimate genetic covariance and correlation from GWAS summary statistics.

Contact
-------

Brendan Bulik-Sullivan, bulik@broadinstitute.org

Hilary Finucane, hilaryf@mit.edu

Citations
---------

For now, please cite

Bulik-Sullivan, et al. LD Score Regression Distinguishes Confounding from Polygenicity in Genome-Wide Association Studies.
In Press at Nature Genetics.

(preprint: http://biorxiv.org/content/early/2014/02/21/002931)

We are currently preparing manuscripts describing the methods for estimating partitioned h2 and rg.

Requirements
------------

1. Python 2.7
2. argparse 1.2.1
3. bitarray 0.8.1
4. numpy 1.8.0
5. pandas 0.15.0
6. scipy 0.10.1

