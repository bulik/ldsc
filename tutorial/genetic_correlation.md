#Genetic Correlation Tutorial

##Overview

This tutorial will walk you through computing the genetic correlation between schizophrenia and bipolar disorder using `ldsc` and the summary statistics from the 2013 [PGC Cross-Disorder paper in the Lancet](http://www.ncbi.nlm.nih.gov/pubmed/23453885). 

This tutorial requires downloading about 75 MB of data.

## TL;DR

If you want to compute the genetic correlation between schizophrenia and bipolar disorder, type these commands

	$ wget www.med.unc.edu/pgc/files/resultfiles/pgc.cross.bip.zip
	$ wget www.med.unc.edu/pgc/files/resultfiles/pgc.cross.scz.zip
	$ unzip pgc.cross.bip.zip
	$ unzip pgc.cross.scz.zip
	$ mv pgc.cross.bip/pgc.cross.BIP11.2013-05.txt .
	$ mv pgc.cross.scz/pgc.cross.SCZ17.2013-05.txt .
	$ python munge_sumstats.py --sumstats pgc.cross.SCZ17.2013-05.txt --N 17115 --out scz
	$ python munge_sumstats.py --sumstats pgc.cross.BIP11.2013-05.txt --N 11810 --out bip
	$ python ldsc.py --rg scz.sumstats.gz,bip.sumstats.gz --ref-ld-chr ld/ --w-ld-chr ld/ 
		--out scz_bip


##LD Scores

This tutorial assumes that you have already computed the LD Scores that you want to use for LD Score regression. If you just want to estimate genetic correlation in European GWAS, there is probably no need for you to compute your own LD Scores, so you can skip the tutorial on LD Score estimation and download pre-computed LD Scores from URL_GOES_HERE. These LD Scores were computed using 1000 Genomes European data and are appropriate for use with European GWAS data. For GWAS from other populations, you will need to compute population-appropriate LD Scores. 

For the purpose of this tutorial, we assume that you have placed the LD Score files into a director called `ld/`, containing files `*.l2.M_5_50`, `*.l2.ldscore.bz2`.

##Downloading Summary Statistics

First, download summary statistics for schizophrenia (scz) and bipolar disorder (bip) from the [psychiatric genomics consortium website](http://www.med.unc.edu/pgc/downloads). If your machine has the `wget` utility, you can use the commands

	$ wget www.med.unc.edu/pgc/files/resultfiles/pgc.cross.bip.zip
	$ wget www.med.unc.edu/pgc/files/resultfiles/pgc.cross.scz.zip

If your machine does not have `wget`, you can just copy and paste the download urls into your browser. This will yield two files, named `pgc.cross.bip.zip` and `pgc.cross.scz.zip`. Unzip these files either by double-clicking in a GUI, or with the `unzip` command from the command line. 
This will yield directories called `pgc.cross.bip` and `pgc.cross.scz`. Move the summary statistics from these directories to you current working directory with the commands

	$ mv pgc.cross.bip/pgc.cross.BIP11.2013-05.txt .
	$ mv pgc.cross.scz/pgc.cross.SCZ17.2013-05.txt .

The first few lines of the bip file should look like this:

	$ head pgc.cross.SCZ17.2013-05.txt

	snpid hg18chr bp a1 a2 or se pval info ngt CEUaf
	rs3131972	1	742584	A	G	1.092	0.0817	0.2819	0.694	0	0.16055
	rs3131969	1	744045	A	G	1.087	0.0781	0.2855	0.939	0	0.133028
	rs3131967	1	744197	T	C	1.093	0.0835	0.2859	0.869	0	.
	rs1048488	1	750775	T	C	0.9158	0.0817	0.2817	0.694	0	0.836449
	rs12562034	1	758311	A	G	0.9391	0.0807	0.4362	0.977	0	0.0925926
	rs4040617	1	769185	A	G	0.9205	0.0777	0.2864	0.98	0	0.87156
	rs28576697	1	860508	T	C	1.079	0.2305	0.7423	0.123	0	0.74537
	rs1110052	1	863421	T	G	1.088	0.2209	0.702	0.137	0	0.752294
	rs7523549	1	869180	T	C	1.823	0.8756	0.4929	0.13	0	0.0137615
	
The first few lines of the scz file should look like this 

	$ head pgc.cross.SCZ17.2013-05.txt

	snpid hg18chr bp a1 a2 or se pval info ngt CEUaf
	rs3131972	1	742584	A	G	1	0.0966	0.9991	0.702	0	0.16055
	rs3131969	1	744045	A	G	1	0.0925	0.9974	0.938	0	0.133028
	rs3131967	1	744197	T	C	1.001	0.0991	0.9928	0.866	0	.
	rs1048488	1	750775	T	C	0.9999	0.0966	0.9991	0.702	0	0.836449
	rs12562034	1	758311	A	G	1.025	0.0843	0.7716	0.988	0	0.0925926
	rs4040617	1	769185	A	G	0.9993	0.092	0.994	0.979	0	0.87156
	rs4970383	1	828418	A	C	1.096	0.1664	0.5806	0.439	0	0.201835
	rs4475691	1	836671	T	C	1.059	0.1181	0.6257	1.02	0	0.146789
	rs1806509	1	843817	A	C	0.9462	0.1539	0.7193	0.383	0	0.600917

##Reformatting Summary Statistics

The summary statistics are not in the `.sumstats` format (defined in the [docs](../docs/file_formats_sumstats.txt)) that `ldsc` understands. We strongly recommend that you use the script `munge_sumstats.py` included in this github repository in order to convert summary statistics into the `ldsc` format, because this script checks for a lot of annoying gotchas that have gotten us in trouble before. 

The `ldsc` `.sumstats` format requires five pieces of information for each SNP:

1. A unique identifier (e.g., the rs number)
2. Allele 1
3. Allele 2
4. Sample size (which often varies from SNP to SNP)
5. A Z-score 

In addition, because imputation quality is correlated with LD Score, we usually remove poorly-imputed SNPs by filtering on INFO > 0.9. The scz and bip summary statistics that we're using for this tutorial have INFO columns, so `munge_sumstats.py` will automatically perform the filtering. If you're using summary statistics that don't come with an INFO column, we recommend filtering to HapMap3 SNPs (using the `--merge` or `--merge-alleles` flags), because these seem to be well-imputed in most studies.

The two sets of summary statistics that we're using for this tutorial don't have sample size columns, so we'll have to assume that sample size is the same for all SNPs and specify these sample sizes using the `--N` flag. The sample size for the scz study in question was 17115 and the sample size for the bip study was 11810. 

To convert the summary statistics, type the commands

	$ python munge_sumstats.py --sumstats pgc.cross.SCZ17.2013-05.txt --N 17115 --out scz
	$ python munge_sumstats.py --sumstats pgc.cross.BIP11.2013-05.txt --N 11810 --out bip

These commands should take about 20 seconds each, though of course the precise time will vary from machine to machine. This will print a series of log messages to the terminal (described below), along with files, `scz.log`, `scz.sumstats.gz` and `bip.log`, `bip.sumstats.gz`. `munge_sumstats.py` will print warning messages labeled `WARNING` to the log file if it finds anything troubling. You can and should search your log files for warnings with the command `grep 'WARNING' *log`. It turns out there are no warnings for these data. 

## Reading the Log Files

I'll describe the contents of the log file section-by-section.

The first section is just the `ldsc` masthead:

	\**********************************************************************
	* LD Score Regression (LDSC)
	* Version 1.0.0
	* (C) 2014-2015 Brendan Bulik-Sullivan and Hilary Finucane
	* Broad Institute of MIT and Harvard / MIT Department of Mathematics
	* GNU General Public License v3
	\**********************************************************************

The next section tells you what command line options you entered. This section is useful for when you're looking at old log files, wondering precisely how some data were processed.

	Options:
	--out scz
	--N 17115.0
	--sumstats data/pgc.cross.SCZ17.2013-05.txt.bz2

The next section describes how `munge_sumstats.py` interprets the column headers. `munge_sumstats.py` can understand most column headers by default, but if your summary statistics have exotic column names, you may need to tell `munge_sumstats.py` what it should do. For example, if the `foobar` column contains INFO scores, you should type `munge_sumstats.py --INFO foobar`. You should always check this section of the log file to make sure that `munge_sumstats.py` understood your column headers correctly. If you are not sure whether `munge_sumstats.py` will understand your column headers, the easiest thing to do is just run `munge_sumstats.py`; if it does not understand the column headers, it will raise an error immediately. 

	Interpreting column names as follows:
	info:   INFO score (imputation quality; higher --> better imputation)
	snpid:  Variant ID (e.g., rs number)
	a1:     Allele 1
	pval:   p-Value
	a2:     Allele 2
	or:     Odds ratio (1 --> no effect; above 1 --> A1 is risk increasing)

This section describes the filtering process. By default, `munge_sumstats.py` filters on INFO > 0.9, MAF > 0.01 and 0 < P <= 1. It also removes variants that are not SNPs (e.g., indels), strand ambiguous SNPs, and SNPs with duplicated rs numbers. If there is an N column (sample size), it removes SNPs with low values of N. Finally, `munge_sumstats.py` checks that the median value of the signed summary statistic column (beta, Z, OR, log OR) is close to the null median (e.g., median OR should be close to 1) in order to make sure that this column is not mislabeled (it is surprisingly common for columns labeled OR to contain log odds ratios).

	Reading sumstats from data/pgc.cross.SCZ17.2013-05.txt.bz2 into memory 5000000.0 SNPs at a time.
	Read 1237958 SNPs from --sumstats file.
	Removed 0 SNPs with missing values.
	Removed 303707 SNPs with INFO <= 0.9.
	Removed 0 SNPs with MAF <= 0.01.
	Removed 0 SNPs with out-of-bounds p-values.
	Removed 0 variants that were not SNPs or were strand-ambiguous.
	856732 SNPs remain.
	Removed 0 SNPs with duplicated rs numbers (856732 SNPs remain).
	Using N = 17115.0
	Median value of or was 1.0, which seems sensible.
	Writing summary statistics for 856732 SNPs (856732 with nonmissing beta) to scz.sumstats.gz.

The last section shows some basic metadata about the summary statistics. If mean chi-square is below 1.02, `munge_sumstats.py` will warn you that the data probably are not suitable for LD Score regression.

	Metadata:
	Mean chi^2 = 1.243
	Lambda GC = 1.205
	Max chi^2 = 32.847
	23 Genome-wide significant SNPs (some may have been removed by filtering).

	Conversion finished at Wed Jan 28 18:17:08 2015
	Total time elapsed: 17.38s 


## Estimating Genetic Correlation

Now that we have all the files that we need in the correct format, we can run LD Score regression with the following command:

	$ python ldsc.py --rg scz.sumstats.gz,bip.sumstats.gz --ref-ld-chr ld/ --w-ld-chr ld/ 
		--out scz_bip

This will take about a minute, though the precise time will of course vary from machine to machine. Let's walk through the components of this command. 
###### `--rg`
The `--rg` flag tells `ldsc` to compute genetic correlation. The argument to `--rg` should be a comma-separated list of files in the `.sumstats` format. In this case, we have only passed two files to `--rg`, but if we were to pass three or more files, `ldsc.py` would compute the genetic correlation between the first file and the list and all subsequent files (i.e., --rg a,b,c will compute rg(a,b) and rg(a,c) ). 
###### `--ref-ld-chr`
The `--ref-ld` flag tells `ldsc` which LD Score files to use as the independent variable in the LD Score regression. The `--ref-ld-chr` flag is used for LD Score files split across chromosomes. By default, `ldsc` appends the chromosome number to the end. For example, typing `--ref-ld-chr ld/` tells `ldsc` to use the files `ld/1.l2.ldscore, ... , ld/22.l2.ldscore`. If the chromosome number is in the middle of the filename, you can tell `ldsc` where to insert the chromosome number by using an `@` symbol. For example, `--ref-ld-chr ld/chr@`.  The argument to `--ref-ld` should omit the `.l2.ldscore` or `.l2.ldscore.gz` file suffix.
###### `--w-ld-chr`
The `--w-ld` flag tells `ldsc` which LD Scores to use for the regression weights. Ideally, these should be $\ell_j^w:=\sum_{j\in reg} r^2_{jk}$, where $reg$ is the set of SNPs included in the regression. However, for this tutorial, we are using the same set of LD Scores for `--w-ld` and `--ref-ld`. In practice, LD Score regression is not very sensitive to the precise choice of LD Scores used for the `--w-ld` flag. For example, if you want to compute genetic correlation between scz and bip with 850,000 regression SNPs and genetic correlation between scz and major depression with (say) 840,000 regression SNPs, almost all of which are overlapping, then you should save time and use the same `--w-ld-chr` LD Scores for both regressions.

There is also a `--w-ld-chr` flag; the syntax is identical to the `--ref-ld-chr` flag.

###### `--out`
This tells `ldsc` where to print the results. If you set `--out foo_bar`, `ldsc` will print results to `foo_bar.log`. If you do not set the `--out` flag, `ldsc` will default to printing results to `ldsc.log`.


## Reading the Results File

I will describe the log file section-by-section. The basic structure 
1. log messages about reading input files
2. heritability of the first trait (in this case, scz)
3. heritability of the second trait (in this case, bip)
4. genetic covariance
5. genetic correlation
6. table of genetic correlations (this is very useful for `--rg` with more than two traits)

The first section is just the masthead and list of command line options:

	*********************************************************************
	* LD Score Regression (LDSC)
	* Version 1.0.0
	* (C) 2014-2015 Brendan Bulik-Sullivan and Hilary Finucane
	* Broad Institute of MIT and Harvard / MIT Department of Mathematics
	* GNU General Public License v3
	*********************************************************************

	Options:
	--ref-ld-chr data/
	--out test
	--rg scz.sumstats.gz,bip.sumstats.gz
	--w-ld-chr data/

The next section shows some basic log messages about reading and merging LD Scores and summary statistics. This section isn't that interesting; the only thing to check is whether the number of SNPs drops unexpectedly at any stage. If this happens, it can indicate a data munging error (e.g., mismatched rs numbers). If the number of SNPs is below 200,000, this is usually bad, and `ldsc` will print a warning.

	Beginning analysis at Thu Jan 29 19:11:21 2015
	Reading summary statistics from scz.sumstats.gz ...
	Read summary statistics for 856732 SNPs.
	Reading reference panel LD Score from data/[1-22] ...
	Read reference panel LD Scores for 1293150 SNPs.
	Reading regression weight LD Score from data/[1-22] ...
	Read regression weight LD Scores for 1293150 SNPs.
	After merging with reference panel LD, 840504 SNPs remain.
	After merging with regression SNP LD, 840504 SNPs remain.
	Computing rg for phenotype 2/2
	Reading summary statistics from bip.sumstats.gz ...
	Read summary statistics for 829068 SNPs.
	After merging with summary staistics, 803425 SNPs remain.
	803425 SNPs with valid alleles.

The next two sections show the heritabilities of each trait from single-trait LD Score regression. These estimates will be biased downwards by GC correction. Note that these heritability estimates are on the observed scale. Lambda GC is $\mathrm{median}(\chi^2)/0.4549$. Mean chi^2 is the mean chi-square statistic. Intercept is the LD Score regression intercept. The intercept should be close to 1, unless the data have been GC corrected, in which case it will often be lower. Ratio is $(\mathrm{intercept}-1)/(\mathrm{mean}(\chi^2)-1)$, which measures the proportion of the inflation in the mean chi^2 that the LD Score regression intercept ascribes to causes other than polygenic heritability. The value of ratio should be close to zero, though in practice values of 10-20% are not uncommon, probably due to sample/reference LD Score mismatch or model misspecification (e.g., low LD variants have slightly higher $h^2$ per SNP)

	Heritability of phenotype 1
	---------------------------
	Total Observed scale h2: 0.5909 (0.0484)
	Lambda GC: 1.2054
	Mean Chi^2: 1.2336
	Intercept: 1.0013 (0.0112)
	Ratio: 0.0057 (0.0481)

	Heritability of phenotype 2/2
	-----------------------------
	Total Observed scale h2: 0.3604 (0.0367)
	Lambda GC: 1.1382
	Mean Chi^2: 1.1437
	Intercept: 1.0013 (0.0093)
	Ratio: 0.0092 (0.065)

The next section shows the genetic covariance. Genetic covariance will be biased downwards by GC correction. The intercept is shown on the same scale as the single-trait LD Score regression intercept. Multiply by $\sqrt{N_1N_2}$ in order to obtain an intercept on the $N_s\rho$ scale. 

	Genetic Covariance
	------------------
	Total Observed scale gencov: 0.3639 (0.037)
	Mean z1*z2: 0.1226
	Intercept: 0.0038 (0.0072)

The next section shows the genetic correlation, Z-score and P-value. The genetic correlation estimate is not biased by GC correction.

	Genetic Correlation
	-------------------
	Genetic Correlation: 0.7885 (0.0726)
	Z-score: 10.8621
	P: 1.7478e-27


The last section (which may not fit too well on your screen) is a table summarizing all results. This is particularly useful when running `--rg` with more than two traits. The columns are p1 = trait 1, p2 = trait 2, rg = genetic correlation, se = standard error of rg, p = p-value for rg; h2_obs = observed scale h2 for trait 2, h2_obs_se = standard error of observed scale h2 for trait 2, h2_int = single-trait LD Score regression intercept for trait 2, h2_int_se = single trait LD Score regression intercept standard error for trait 2, gcov_int = cross-trait LD Score regression intercept, gcov_int_se = cross-trait LD Score regression intercept standard error.

	Summary of Genetic Correlation Results
		              p1               p2     rg     se          p  h2_obs h2_obs_se  h2_int  h2_int_se  gcov_int  gcov_int_se
	 scz.sumstats.gz  bip.sumstats.gz  0.789  0.073  1.748e-27    0.36        NA   1.001      0.009     0.004        0.007

	Analysis finished at Thu Jan 29 19:11:49 2015
	Total time elapsed: 28.0s
	
## Conversion to Liability Scale

There is no notion of observed or liability scale genetic correlation. We can compute genetic correlation between pairs of quantitative traits, one quantitative trait and one binary trait, and pairs of binary traits without having to worry about different scales. In addition, if we compute genetic correlations from two studies of the same binary trait with different sample prevalences, we should expect to get the same result, modulo noise.

For heritability and genetic covariance, it is customary to report heritability on the liability scale, because liability scale heritability is comparable across studies with different prevalences. By default, `ldsc` outputs observed scale heritability. To convert to the liability scale, we need to tell `ldsc` the sample and population prevalence for each trait using the `--samp-prev` and `--pop-prev` flags, respectively. The population prevalence of scz and bip are both around 1%, and the sample prevalence in each of these studies was about 50%, so 

	$ python ldsc.py --rg scz.sumstats.gz,bip.sumstats.gz --ref-ld-chr ld/ --w-ld-chr ld/ 
		--out scz_bip --samp-prev 0.5,0.5 --pop-prev 0.01,0.01

The output is the same as before, except 'Observed' is replaced with 'Liability'.

If you're computing genetic covariance between one binary trait and one quantitative trait, then you can tell `ldsc` that (say) the second trait is a quantitative trait via `--samp-prev 0.5,nan --pop-prev 0.01,nan`.