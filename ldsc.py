'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

LDSC is a command line tool for estimating
	1. LD Score 
	2. heritability / partitioned heritability
	3. genetic covariance / correlation
	
'''
from __future__ import division
import ldscore.ldscore as ld
import ldscore.parse as ps
import ldscore.jackknife as jk
import ldscore.sumstats as sumstats
import numpy as np
import pandas as pd
from subprocess import call
from itertools import product
import time, sys, traceback, argparse

__version__ = '1.0.0'
MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* LD Score Regression (LDSC)\n"
MASTHEAD += "* Version {V}\n".format(V=__version__)
MASTHEAD += "* (C) 2014-2015 Brendan Bulik-Sullivan and Hilary Finucane\n"
MASTHEAD += "* Broad Institute of MIT and Harvard / MIT Department of Mathematics\n"
MASTHEAD += "* GNU General Public License v3\n"
MASTHEAD += "*********************************************************************\n"
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
pd.set_option('max_colwidth',1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)

def sec_to_str(t):
	'''Convert seconds to days:hours:minutes:seconds'''
	[d, h, m, s, n] = reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(t, 1), 60,60, 24])
	f = ''
	if d > 0:
		f += '{D}d:'.format(D=d)
	if h > 0:
		f += '{H}h:'.format(H=h)
	if m > 0: 
		f += '{M}m:'.format(M=m)
	
	f += '{S}s'.format(S=s)
	return f

def _remove_dtype(x):
	'''Removes dtype: float64 and dtype: int64 from pandas printouts'''
	x = str(x)
	x = x.replace('\ndtype: int64','')
	x = x.replace('\ndtype: float64','')
	return x
	
class Logger(object):
	'''
	Lightweight logging.
	TODO: replace with logging module
	
	'''
	def __init__(self, fh):
		self.log_fh = open(fh, 'wb')
		
 	def log(self, msg):
		'''
		Print to log file and stdout with a single command.
		
		'''
		print >>self.log_fh, msg
		print msg
			

def __filter__(fname, noun, verb, merge_obj):
	merged_list = None
	if fname:
		f = lambda x,n: x.format(noun=noun, verb=verb, fname=fname, num=n)
		x = ps.FilterFile(fname)
	 	c = 'Read list of {num} {noun} to {verb} from {fname}'
	 	print f(c, len(x.IDList))
		merged_list = merge_obj.loj(x.IDList)
		len_merged_list = len(merged_list)
		if len_merged_list > 0:
			c = 'After merging, {num} {noun} remain'
			print f(c, len_merged_list)
		else:
			error_msg = 'No {noun} retained for analysis'
			raise ValueError(f(error_msg, 0))

		return merged_list
		
def annot_sort_key(s):
	'''For use with --cts-bin. Fixes weird pandas crosstab column order.'''
	if type(s) == tuple:
		s = [x.split('_')[0] for x in s]
		s = map(lambda x: float(x) if x != 'min' else -float('inf'), s)
	else: #type(s) = str:	
		s = s.split('_')[0]
		if s == 'min': 
			s = float('-inf')
		else:
			s = float(s)
				
 	return s

def ldscore(args, header=None):
	'''
	Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
	reference panel genotypes. 
	
	Annot format is 
	chr snp bp cm <annotations>
	
	'''
	log = logger(args.out+'.log')
	if header:
		log.log(header)

	if args.bfile:
		snp_file, snp_obj = args.bfile+'.bim', ps.PlinkBIMFile
		ind_file, ind_obj = args.bfile+'.fam', ps.PlinkFAMFile
		array_file, array_obj = args.bfile+'.bed', ld.PlinkBEDFile

	# read bim/snp
	array_snps = snp_obj(snp_file)
	m = len(array_snps.IDList)
	log.log('Read list of {m} SNPs from {f}'.format(m=m, f=snp_file))

	# read --annot
	if args.annot is not None:
		annot = ps.AnnotFile(args.annot)
		n_annot, ma = len(annot.df.columns) - 4, len(annot.df)
		log.log("Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,
			A=n_annot, M=ma))
		annot_matrix = np.array(annot.df.iloc[:,4:])
		annot_colnames = annot.df.columns[4:]
		keep_snps = None
		if np.any(annot.df.SNP.values != array_snps.df.SNP.values):
			raise ValueError('The .annot file must contain the same SNPs in the same'+\
				' order as the .bim file.')
	# read --extract
	elif args.extract is not None:
		keep_snps = __filter__(args.extract, 'SNPs', 'include', array_snps)
		annot_matrix, annot_colnames, n_annot = None, None, 1
	
	# read --cts-bin plus --cts-breaks
	elif args.cts_bin is not None and args.cts_breaks is not None:
		# read filenames
		cts_fnames = args.cts_bin.split(',')
		# read breaks
		# replace N with negative sign
		args.cts_breaks = args.cts_breaks.replace('N','-')
		# split on x
		try:
			breaks = [[float(x) for x in y.split(',')] for y in args.cts_breaks.split('x')]
		except ValueError as e:
			raise ValueError('--cts-breaks must be a comma-separated list of numbers: '
				+str(e.args))
	
		if len(breaks) != len(cts_fnames):
			raise ValueError('Need to specify one set of breaks for each file in --cts-bin.')
		
		if args.cts_names:
			cts_colnames = [str(x) for x in args.cts_names.split(',')]
			if len(cts_colnames) != len(cts_fnames):
				msg = 'Must specify either no --cts-names or one value for each file in --cts-bin.'
				raise ValueError(msg)

		else:
			cts_colnames = ['ANNOT'+str(i) for i in xrange(len(cts_fnames))]
			
		log.log('Reading numbers with which to bin SNPs from {F}'.format(F=args.cts_bin))
	
		cts_levs = []
		full_labs = []
		for i,fh in enumerate(cts_fnames):
			vec = ps.read_cts(cts_fnames[i], array_snps.df.SNP.values)
			
			max_cts = np.max(vec)
			min_cts = np.min(vec)
			cut_breaks = list(breaks[i])
			name_breaks = list(cut_breaks)
			if np.all(cut_breaks >= max_cts) or np.all(cut_breaks <= min_cts):
				raise ValueError('All breaks lie outside the range of the cts variable.')

			if np.all(cut_breaks <= max_cts):
				name_breaks.append(max_cts)
				cut_breaks.append(max_cts+1)
		
			if np.all(cut_breaks >= min_cts):	
				name_breaks.append(min_cts)
				cut_breaks.append(min_cts-1)

			name_breaks.sort()
			cut_breaks.sort()		
			n_breaks = len(cut_breaks)
			# so that col names are consistent across chromosomes with different max vals
			name_breaks[0] = 'min'
			name_breaks[-1] = 'max'
			name_breaks = [str(x) for x in name_breaks]
			labs = [name_breaks[i]+'_'+name_breaks[i+1] for i in xrange(n_breaks-1)]
			cut_vec = pd.Series(pd.cut(vec, bins=cut_breaks, labels=labs))
			cts_levs.append(cut_vec)
			full_labs.append(labs)

		annot_matrix = pd.concat(cts_levs, axis=1)
		annot_matrix.columns = cts_colnames
		# crosstab -- for now we keep empty columns
		annot_matrix = pd.crosstab(annot_matrix.index, 
			[annot_matrix[i] for i in annot_matrix.columns], dropna=False,
			colnames=annot_matrix.columns)

		# add missing columns
		if len(cts_colnames) > 1:
			for x in product(*full_labs):
				if x not in annot_matrix.columns:
					annot_matrix[x] = 0
		else:
			for x in full_labs[0]:
				if x not in annot_matrix.columns:
					annot_matrix[x] = 0
				
		annot_matrix = annot_matrix[sorted(annot_matrix.columns, key=annot_sort_key)]
		if len(cts_colnames) > 1:
			# flatten multi-index
			annot_colnames = ['_'.join([cts_colnames[i]+'_'+b for i,b in enumerate(c)])
				for c in annot_matrix.columns]
		else:
			annot_colnames = [cts_colnames[0]+'_'+b for b in annot_matrix.columns]

		annot_matrix = np.matrix(annot_matrix)
		keep_snps = None
		n_annot = len(annot_colnames)
		if np.any(np.sum(annot_matrix, axis=1) == 0):
 			# This exception should never be raised. For debugging only.
 			raise ValueError('Some SNPs have no annotation in --cts-bin. This is a bug!')

	else:
		annot_matrix, annot_colnames, keep_snps = None, None, None, 
		n_annot = 1
	
	# read fam
	array_indivs = ind_obj(ind_file)
	n = len(array_indivs.IDList)	 
	log.log('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))
	# read keep_indivs
	if args.keep:
		keep_indivs = __filter__(args.keep, 'individuals', 'include', array_indivs)
	else:
		keep_indivs = None
	
	# read genotype array
	log.log('Reading genotypes from {fname}'.format(fname=array_file))
	geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
		keep_indivs=keep_indivs, mafMin=args.maf)
		
	# filter annot_matrix down to only SNPs passing MAF cutoffs
	if annot_matrix is not None:
		annot_keep = geno_array.kept_snps
		annot_matrix = annot_matrix[annot_keep,:]
	
	# determine block widths
	x = np.array((args.ld_wind_snps, args.ld_wind_kb, args.ld_wind_cm), dtype=bool)
	if np.sum(x) != 1: 
		raise ValueError('Must specify exactly one --ld-wind option')
	
	if args.ld_wind_snps:
		max_dist = args.ld_wind_snps
		coords = np.array(xrange(geno_array.m))
	elif args.ld_wind_kb:
		max_dist = args.ld_wind_kb*1000
		coords = np.array(array_snps.df['BP'])[geno_array.kept_snps]
	elif args.ld_wind_cm:
		max_dist = args.ld_wind_cm
		coords = np.array(array_snps.df['CM'])[geno_array.kept_snps]

	block_left = ld.getBlockLefts(coords, max_dist)
	if block_left[len(block_left)-1] == 0 and not args.yes_really:
		error_msg = 'Do you really want to compute whole-chomosome LD Score? If so, set the '
		error_msg += '--yes-really flag (warning: it will use a lot of time / memory)'
		raise ValueError(error_msg)

	scale_suffix = ''
	if args.pq_exp is not None:
		log.log('Computing LD with pq ^ {S}.'.format(S=args.pq_exp))
		msg = 'Note that LD Scores with pq raised to a nonzero power are'
		msg += 'not directly comparable to normal LD Scores.'
		log.log(msg)
		scale_suffix = '_S{S}'.format(S=args.pq_exp)
		pq = np.matrix(geno_array.maf*(1-geno_array.maf)).reshape((geno_array.m,1))
		pq = np.power(pq, args.pq_exp)

		if annot_matrix is not None:
			annot_matrix = np.multiply(annot_matrix, pq)
		else:
			annot_matrix = pq
	
	elif args.maf_exp is not None:
		log.log('Computing LD with MAF ^ {S}.'.format(S=args.maf_exp))
		msg = 'Note that LD Scores with MAF raised to a nonzero power are'
		msg += 'not directly comparable to normal LD Scores.'
		log.log(msg)
		scale_suffix = '_S{S}'.format(S=args.maf_exp)
		mf = np.matrix(geno_array.maf).reshape((geno_array.m,1))
		mf = np.power(mf, args.maf_exp)

		if annot_matrix is not None:
			annot_matrix = np.multiply(annot_matrix, mf)
		else:
			annot_matrix = mf
		
	log.log("Estimating LD Score.")
	lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
	col_prefix = "L2"; file_suffix = "l2"

	if n_annot == 1:
		ldscore_colnames = [col_prefix+scale_suffix]
	else:
		ldscore_colnames =  [x+col_prefix+scale_suffix for x in annot_colnames]
			
	# print .ldscore. Output columns: CHR, BP, CM, RS, MAF, [LD Scores]
	out_fname = args.out + '.' + file_suffix + '.ldscore'
	new_colnames = geno_array.colnames + ldscore_colnames
	df = pd.DataFrame.from_records(np.c_[geno_array.df, lN])
	df.columns = new_colnames
	if args.print_snps:
		if args.print_snps.endswith('gz'):
			print_snps = pd.read_csv(args.print_snps, header=None, compression='gzip')
		elif args.print_snps.endswith('bz2'):
			print_snps = pd.read_csv(args.print_snps, header=None, compression='bz2')
		else:
			print_snps = pd.read_csv(args.print_snps, header=None)
		if len(print_snps.columns) > 1:
			raise ValueError('--print-snps must refer to a file with a one column of SNP IDs.')
		log.log('Reading list of {N} SNPs for which to print LD Scores from {F}'.format(\
						F=args.print_snps, N=len(print_snps)))

		print_snps.columns=['SNP']
		df = df.ix[df.SNP.isin(print_snps.SNP),:]
		if len(df) == 0:
			raise ValueError('After merging with --print-snps, no SNPs remain.')
		else:
			msg = 'After merging with --print-snps, LD Scores for {N} SNPs will be printed.'
			log.log(msg.format(N=len(df)))
	
	if not args.pickle:
		l2_suffix = '.gz'
		log.log("Writing LD Scores for {N} SNPs to {f}.gz".format(f=out_fname, N=len(df)))
		df.to_csv(out_fname, sep="\t", header=True, index=False)	
		call(['gzip', '-f', out_fname])
	elif args.pickle:
		l2_suffix = '.pickle'
		log.log("Writing LD Scores for {N} SNPs to {f}.pickle".format(f=out_fname, N=len(df)))
		df.set_index('SNP')
		out_fname_pickle = out_fname+l2_suffix
		df.reset_index(drop=True).to_pickle(out_fname_pickle)
		
	# print .M
	if annot_matrix is not None:
		M = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix, axis=0))))
		ii = geno_array.maf > 0.05
		M_5_50 = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix[ii,:], axis=0))))
	else:
		M = [geno_array.m]
		M_5_50 = [np.sum(geno_array.maf > 0.05)]
	
	# print .M
	fout_M = open(args.out + '.'+ file_suffix +'.M','wb')
	print >>fout_M, '\t'.join(map(str,M))
	fout_M.close()
	
	# print .M_5_50
	fout_M_5_50 = open(args.out + '.'+ file_suffix +'.M_5_50','wb')
	print >>fout_M_5_50, '\t'.join(map(str,M_5_50))
	fout_M_5_50.close()
	
	# print annot matrix
	if (args.cts_bin is not None or args.cts_bin_add is not None) and not args.no_print_annot:
		out_fname_annot = args.out + '.annot'
		new_colnames = geno_array.colnames + ldscore_colnames
		annot_df = pd.DataFrame(np.c_[geno_array.df, annot_matrix])
		annot_df.columns = new_colnames	
		del annot_df['MAF']
		log.log("Writing annot matrix produced by --cts-bin to {F}".format(F=out_fname+'.gz'))
		if args.gzip:
			annot_df.to_csv(out_fname_annot, sep="\t", header=True, index=False)	
			call(['gzip', '-f', out_fname_annot])
		else:
			out_fname_annot_pickle = out_fname_annot + '.pickle'
			annot_df.reset_index(drop=True).to_pickle(out_fname_annot_pickle)
			
	# print LD Score summary	
	pd.set_option('display.max_rows', 200)
	log.log('\nSummary of LD Scores in {F}'.format(F=out_fname+l2_suffix))
	t = df.ix[:,4:].describe()
	log.log( t.ix[1:,:] )
	
	# print correlation matrix including all LD Scores and sample MAF
	log.log('')
	log.log('MAF/LD Score Correlation Matrix')
	log.log( df.ix[:,4:].corr() )
	
	# print condition number
	if n_annot > 1: # condition number of a column vector w/ nonzero var is trivially one
		log.log('\nLD Score Matrix Condition Number')
		cond_num = np.linalg.cond(df.ix[:,5:])
		log.log( jk.kill_brackets(str(np.matrix(cond_num))) )
		if cond_num > 10000:
			log.log('WARNING: ill-conditioned LD Score Matrix!')
		
	# summarize annot matrix if there is one
	
	if annot_matrix is not None:
		# covariance matrix
		x = pd.DataFrame(annot_matrix, columns=annot_colnames)
		log.log('\nAnnotation Correlation Matrix')
		log.log( x.corr() )

		# column sums
		log.log('\nAnnotation Matrix Column Sums')
		log.log(_remove_dtype(x.sum(axis=0)))
			
		# row sums 
		log.log('\nSummary of Annotation Matrix Row Sums')
		row_sums = x.sum(axis=1).describe()
		log.log(_remove_dtype(row_sums))

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='ldsc', type=str,
	help='Output filename prefix. If --out is not set, LDSC will use ldsc as the '
	'defualt output filename prefix.')
	
# Basic LD Score Estimation Flags'
parser.add_argument('--bfile', default=None, type=str, 
	help='Prefix for Plink .bed/.bim/.fam file')

# Filtering / Data Management for LD Score
parser.add_argument('--extract', default=None, type=str, 
	help='File with SNPs to include in LD Score estimation. '
	'The file should contain one SNP ID per row.')
parser.add_argument('--keep', default=None, type=str, 
	help='File with individuals to include in LD Score estimation. '
	'The file should contain one individual ID per row.')
parser.add_argument('--ld-wind-snps', default=None, type=int,
	help='Specify the window size to be used for estimating LD Scores in units of '
	'# of SNPs. You can only specify one --ld-wind-* option.')
parser.add_argument('--ld-wind-kb', default=None, type=float,
	help='Specify the window size to be used for estimating LD Scores in units of '
	'kilobase-pairs (kb). You can only specify one --ld-wind-* option.')
parser.add_argument('--ld-wind-cm', default=None, type=float,
	help='Specify the window size to be used for estimating LD Scores in units of '
	'centiMorgans (cM). You can only specify one --ld-wind-* option.')
parser.add_argument('--print-snps', default=None, type=str,
	help='This flag tells LDSC to only print LD Scores for the SNPs listed '
	'(one ID per row) in PRINT_SNPS. The sum r^2 will still include SNPs not in '
	'PRINT_SNPs. This is useful for reducing the number of LD Scores that have to be '
	'read into memory when estimating h2 or rg.' )

# Fancy LD Score Estimation Flags
parser.add_argument('--annot', default=None, type=str, 
	help='Filename prefix for annotation file for partitioned LD Score estimation. '
	'LDSC will automatically append .annot or .annot.gz to the filename prefix. '
	'See docs/file_formats_ld for a definition of the .annot format.')
parser.add_argument('--cts-bin', default=None, type=str, 
	help='This flag tells LDSC to compute partitioned LD Scores, where the partition '
	'is defined by cutting one or several continuous variable[s] into bins. '
	'The argument to this flag should be the name of a single file or a comma-separated '
	'list of files. The file format is two columns, with SNP IDs in the first column '
	'and the continuous variable in the second column. ')
parser.add_argument('--cts-breaks', default=None, type=str, 
	help='Use this flag to specify names for the continuous variables cut into bins '
	'with --cts-bin. For each continuous variable, specify breaks as a comma-separated '
	'list of breakpoints, and separate the breakpoints for each variable with an x. '
	'For example, if binning on MAF and distance to gene (in kb), '
	'you might set --cts-breaks 0.1,0.25,0.4x10,100,1000 ')	
parser.add_argument('--cts-names', default=None, type=str, 
	help='Use this flag to specify names for the continuous variables cut into bins '
	'with --cts-bin. The argument to this flag should be a comma-separated list of '
	'names. For example, if binning on DAF and distance to gene, you might set '		
	'--cts-bin DAF,DIST_TO_GENE '
	)
parser.add_argument('--per-allele', default=False, action='store_true',
	help='Setting this flag causes LDSC to compute per-allele LD Scores, '
	'i.e., \ell_j := \sum_k p_k(1-p_k)r^2_{jk}, where p_k denotes the MAF '
	'of SNP j. ')
parser.add_argument('--pq-exp', default=None, type=float,
	help='Setting this flag causes LDSC to compute LD Scores with the given scale factor, '
	'i.e., \ell_j := \sum_k (p_k(1-p_k))^a r^2_{jk}, where p_k denotes the MAF '
	'of SNP j and a is the argument to --pq-exp. ')
parser.add_argument('--no-print-annot', default=False, action='store_true',
	help='By defualt, seting --cts-bin or --cts-bin-add causes LDSC to print '
	'the resulting annot matrix. Setting --no-print-annot tells LDSC not '
	'to print the annot matrix. ')
parser.add_argument('--maf', default=None, type=float,
	help='Minor allele frequency lower bound. Default is MAF > 0.')
	
# Basic Flags for Working with Variance Components
parser.add_argument('--h2', default=None, type=str,
	help='Filename prefix for a .chisq file for one-phenotype LD Score regression. '
	'LDSC will automatically append .chisq or .chisq.gz to the filename prefix.'
	'--h2 requires at minimum also setting the --ref-ld and --w-ld flags.')
parser.add_argument('--rg', default=None, type=str,
	help='Comma-separated list of prefixes of .chisq filed for genetic correlation estimation.')
parser.add_argument('--ref-ld', default=None, type=str,
	help='Use --ref-ld to tell LDSC which LD Scores to use as the predictors in the LD '
	'Score regression. '
	'LDSC will automatically append .l2.ldscore/.l2.ldscore.gz to the filename prefix.')
parser.add_argument('--ref-ld-chr', default=None, type=str,
	help='Same as --ref-ld, but will automatically concatenate .l2.ldscore files split '
	'across 22 chromosomes. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz '
	'to the filename prefix. If the filename prefix contains the symbol @, LDSC will '
	'replace the @ symbol with chromosome numbers. Otherwise, LDSC will append chromosome '
	'numbers to the end of the filename prefix.'
	'Example 1: --ref-ld-chr ld/ will read ld/1.l2.ldscore.gz ... ld/22.l2.ldscore.gz'
	'Example 2: --ref-ld-chr ld/@_kg will read ld/1_kg.l2.ldscore.gz ... ld/22_kg.l2.ldscore.gz')
parser.add_argument('--w-ld', default=None, type=str,
	help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included '
	'in the regression. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz.')
parser.add_argument('--w-ld-chr', default=None, type=str,
	help='Same as --w-ld, but will read files split into 22 chromosomes in the same '
	'manner as --ref-ld-chr.')
parser.add_argument('--overlap-annot', default=False, action='store_true',
	help='This flag informs LDSC that the partitioned LD Scores were generates using an '
	'annot matrix with overlapping categories (i.e., not all row sums equal 1), '
	'and prevents LDSC from displaying output that is meaningless with overlapping categories.')
parser.add_argument('--no-filter-chisq', default=False, action='store_true',
	help='Don\'t remove SNPs with large chi-square.')
parser.add_argument('--max-chisq', default=None, type=float,
	help='Max chi^2 for SNPs in the regression.')
parser.add_argument('--no-intercept', action='store_true',
	help = 'If used with --h2, this constrains the LD Score regression intercept to equal '
	'1. If used with --rg, this constrains the LD Score regression intercepts for the h2 '
	'estimates to be one and the intercept for the genetic covariance estimate to be zero.')
parser.add_argument('--constrain-intercept', action='store', default=None,
	help = 'If used with --h2, constrain the regression intercept to be a fixed value. '
	'If used with -rg, constrain the regression intercepts to a comma-separated list '
	'of three values, where the first value is the intercept of the first h2 regression, '
	'the second value is the intercept of the second h2 regression, and the third '
	'value is the intercept of the genetic covaraince regression (i.e., an estimate '
	'of (# of overlapping samples)*(phenotpyic correlation). ')
parser.add_argument('--M', default=None, type=str,
	help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')

# Flags for both LD Score estimation and h2/gencor estimation
parser.add_argument('--print-cov', default=False, action='store_true',
	help='For use with --h2/--rg. This flag tells LDSC to print the '
	'covaraince matrix of the estimates.')
# frequency (useful for .bin files)
parser.add_argument('--print-delete-vals', default=False, action='store_true',
	help='If this flag is set, LDSC will print the block jackknife delete-values ('
	'i.e., the regression coefficeints estimated from the data with a block removed). '
	'The delete-values are formatted as a matrix with (# of jackknife blocks) rows and '
	'(# of LD Scores) columns.')
# Flags you should almost never use
parser.add_argument('--chunk-size', default=50, type=int,
	help='Chunk size for LD Score calculation. Use the default.')
parser.add_argument('--pickle', default=False, action='store_true',
	help='Store .l2.ldscore files as pickles instead of gzipped tab-delimited text.')
parser.add_argument('--yes-really', default=False, action='store_true',
	help='Yes, I really want to compute whole-chromosome LD Score.')
parser.add_argument('--invert-anyway', default=False, action='store_true',
	help="Force LDSC to attempt to invert ill-conditioned matrices.")
parser.add_argument('--n-blocks', default=200, type=int,
	help='Number of block jackknife blocks.')
parser.add_argument('--not-M-5-50', default=False, action='store_true',
	help='This flag tells LDSC to use the .l2.M file instead of the .l2.M_5_50 file.')
parser.add_argument('--return-silly-things', default=False, action='store_true',
	help='Force ldsc to return silly genetic correlation estimates.')
parser.add_argument('--no-check-alleles', default=False, action='store_true',
	help='For rg estimation, skip checking whether the alleles match. This check is '
	'redundant for pairs of chisq files generated using munge_sumstats.py and the '
	'same argument to the --merge-alleles flag.')
parser.add_argument('--print-coefficients',default=False,action='store_true',
	help='when categories are overlapping, print coefficients as well as heritabilities.')
parser.add_argument('--frqfile', type=str, 
	help='For use with --overlap-annot. Provides allele frequencies to prune to common '
	'snps if --not-M-5-50 is not set.')

# transform to liability scale
parser.add_argument('--samp-prev',default=None,
	help='Sample prevalence of binary phenotype (for conversion to liability scale).')
parser.add_argument('--pop-prev',default=None,
	help='Population prevalence of binary phenotype (for conversion to liability scale).')

if __name__ == '__main__':

	args = parser.parse_args()
	if args.out is None:
		raise ValueError('--out is required.')

	log = Logger(args.out+'.log')
	try: 
		defaults = vars(parser.parse_args(''))
		opts = vars(args)
		non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]	
		header = MASTHEAD
		header += "\nOptions: \n"
		options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
		header += '\n'.join(options).replace('True','').replace('False','')+'\n'
		log.log(header)
		log.log('Beginning analysis at {T}'.format(T=time.ctime()))
		start_time = time.time()
		if args.no_intercept and args.h2:
			args.constrain_intercept = 1
		elif args.no_intercept and args.rg:
			args.constrain_intercept = '1,1,0'
	
		if args.constrain_intercept:
			args.constrain_intercept = args.constrain_intercept.replace('N','-')
			if args.h2:
				args.constrain_intercept = float(args.h2)
			elif args.rg:
				args.constrain_intercept = [float(x) for x in args.constrain_intercept.split(',')]
	
		if args.n_blocks <= 1:
			raise ValueError('--n-blocks must be an integer > 1.')
		if args.bfile is not None:
			if args.l2 is None:
				raise ValueError('Must specify --l2 with --bfile.')
			if args.annot is not None and args.extract is not None:
				raise ValueError('--annot and --extract are currently incompatible.')
			if args.cts_bin is not None and args.extract is not None:
				raise ValueError('--cts-bin and --extract are currently incompatible.')
			if args.annot is not None and args.cts_bin is not None:
				raise ValueError('--annot and --cts-bin are currently incompatible.')	
			if (args.cts_bin is not None) != (args.cts_breaks is not None):
				raise ValueError('Must set both or neither of --cts-bin and --cts-breaks.')
			if args.per_allele and args.pq_exp is not None:
				raise ValueError('Cannot set both --per-allele and --pq-exp (--per-allele is equivalent to --pq-exp 1).')
			if args.per_allele:
				args.pq_exp = 1
		
			ldscore(args, log)
		# summary statistics
		elif (args.h2 or args.rg) and (args.ref_ld or args.ref_ld_chr) and (args.w_ld or args.w_ld_chr):
			if args.h2 is not None and args.rg is not None:	
				raise ValueError('Cannot set both --h2 and --rg.')
			if args.ref_ld and args.ref_ld_chr:
				raise ValueError('Cannot set both --ref-ld and --ref-ld-chr.')
			if args.w_ld and args.w_ld_chr:
				raise ValueError('Cannot set both --w-ld and --w-ld-chr.')
			if (args.samp_prev is not None) != (args.pop_prev is not None):
				raise ValueError('Must set both or neither of --samp-prev and --pop-prev.') 
			
			if args.rg:
				if args.samp_prev is not None and args.pop_prev is not None:
					if not (',' in args.samp_prev and ',' in args.pop_prev):
						raise ValueError('For --rg, --samp-prev and --pop-prev must each have at least two numbers separated by a comma.')
					else:
						args.samp_prev, args.pop_prev = map(lambda x: map(float, x.split(',')), 
							[args.samp_prev, args.pop_prev])

				sumstats.estimate_rg(args, log)
			elif args.h2:
				if args.samp_prev is not None and args.pop_prev is not None:
					args.samp_prev, args.pop_prev = map(float, [args.samp_prev, args.pop_prev])
				sumstats.estimate_h2(args, log)

			# bad flags
		else:
			print header
			print 'Error: no analysis selected.'
			print 'ldsc.py -h describes options.'
	except Exception:
		ex_type, ex, tb = sys.exc_info()
		log.log( traceback.format_exc(ex) )
		raise
	finally:
		log.log('Analysis finished at {T}'.format(T=time.ctime()) )
		time_elapsed = round(time.time()-start_time,2)
		log.log('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
