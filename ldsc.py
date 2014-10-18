'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This is a command line application for estimating
	1. LD Score and friends (L1, L1^2, L2 and L4)
	2. heritability / partitioned heritability
	3. genetic covariance
	4. genetic correlation
	5. block jackknife standard errors for all of the above.
	
	
'''
from __future__ import division
import ldscore.ldscore as ld
import ldscore.parse as ps
import ldscore.jackknife as jk
import argparse
import numpy as np
import pandas as pd
from subprocess import call
from itertools import product

__version__ = '0.0.1 (alpha)'

MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* LD Score Regression (LDSC)\n"
MASTHEAD += "* Version {V}\n".format(V=__version__)
MASTHEAD += "* (C) 2014 Brendan Bulik-Sullivan and Hilary Finucane\n"
MASTHEAD += "* Broad Institute of MIT and Harvard / MIT Department of Mathematics\n"
MASTHEAD += "* GNU General Public License v3\n"
MASTHEAD += "*********************************************************************\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


def _remove_dtype(x):
	'''Removes dtype: float64 and dtype: int64 from pandas printouts'''
	x = str(x)
	x = x.replace('\ndtype: int64','')
	x = x.replace('\ndtype: float64','')
	return x

class logger(object):
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
		

def _print_cov(hsqhat, ofh, log):
	'''Prints covariance matrix of slopes'''
	log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
	np.savetxt(ofh, hsqhat.hsq_cov)


def _print_gencov_cov(hsqhat, ofh, log):
	'''Prints covariance matrix of slopes'''
	log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
	np.savetxt(ofh, hsqhat.gencov_cov)


def _print_delete_k(hsqhat, ofh, log):
	'''Prints block jackknife delete-k values'''
	log.log('Printing block jackknife delete-k values to {F}'.format(F=ofh))
	out_mat = hsqhat._jknife.delete_values
	if hsqhat.constrain_intercept is None:
		ncol = out_mat.shape[1]
		out_mat = out_mat[:,0:ncol-1]
		
	np.savetxt(ofh, out_mat)

	
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
	#log.log(args)
	
	if args.bin:
		snp_file, snp_obj = args.bin+'.bim', ps.PlinkBIMFile
		ind_file, ind_obj = args.bin+'.ind', ps.VcfINDFile
		array_file, array_obj = args.bin+'.bin', ld.VcfBINFile
	elif args.bfile:
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
		num_annots, ma = len(annot.df.columns) - 4, len(annot.df)
		log.log("Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,
			A=num_annots, M=ma))
		annot_matrix = np.array(annot.df.iloc[:,4:])
		annot_colnames = annot.df.columns[4:]
		keep_snps = None
		if np.any(annot.df.SNP.values != array_snps.df.SNP.values):
			raise ValueError('The .annot file must contain the same SNPs in the same'+\
				' order as the .bim file.')
	# read --extract
	elif args.extract is not None:
		keep_snps = __filter__(args.extract, 'SNPs', 'include', array_snps)
		annot_matrix, annot_colnames, num_annots = None, None, 1
	
	# read cts_bin_add
	elif args.cts_bin_add is not None and args.cts_breaks is not None:
		# read filenames
		cts_fnames = args.cts_bin_add.split(',')
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
			
		log.log('Reading numbers with which to bin SNPs from {F}'.format(F=args.cts_bin_add))
	
		cts_levs = []
		full_labs = []
		first_lev = np.zeros((m,))
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
			full_labs.append(labs)
			small_annot_matrix = cut_vec
			# crosstab -- for now we keep empty columns
			small_annot_matrix = pd.crosstab(small_annot_matrix.index, 
				small_annot_matrix, dropna=False)
			small_annot_matrix = small_annot_matrix[sorted(small_annot_matrix.columns, key=annot_sort_key)]
			cts_levs.append(small_annot_matrix.ix[:,1:])
			# first column defaults to no annotation
			first_lev += small_annot_matrix.ix[:,0]
	
		if len(cts_colnames) == 1:
			annot_colnames = [cts_colnames[0]+'_'+bin for bin in full_labs[0]]
		else:
			annot_colnames = []
			for i,cname in enumerate(cts_colnames):
				for bin in full_labs[i][1:]:
					annot_colnames.append(cts_colnames[i]+'_'+bin)
					
		annot_colnames.insert(0, "BOTTOM_BINS")
		first_lev = np.minimum(first_lev, 1)
		cts_levs.insert(0, pd.DataFrame(first_lev))
		annot_matrix = pd.concat(cts_levs, axis=1)
		annot_matrix = np.matrix(annot_matrix)
		keep_snps = None
		num_annots = annot_matrix.shape[1]

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
		num_annots = len(annot_colnames)
		if np.any(np.sum(annot_matrix, axis=1) == 0):
 			# This exception should never be raised. For debugging only.
 			raise ValueError('Some SNPs have no annotation in --cts-bin. This is a bug!')

	else:
		annot_matrix, annot_colnames, keep_snps = None, None, None, 
		num_annots = 1
	
	# read fam/ind
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
	
# 	if args.se: # block jackknife
# 
# 		# block size
# 		if args.block_size:
# 			jSize = args.block_size 
# 		elif n > 50:
# 			jSize = 10
# 		else:
# 			jSize = 1
# 		
# 		jN = int(np.ceil(n / jSize))
# 		if args.l1:
# 			col_prefix = "L1"; file_suffix = "l1.jknife"
# 			raise NotImplementedError('Sorry, havent implemented L1 block jackknife yet.')
# 			
# 		elif args.l1sq:
# 			col_prefix = "L1SQ"; file_suffix = "l1sq.jknife"
# 			raise NotImplementedError('Sorry, havent implemented L1^2 block jackknife yet.')
# 			
# 		elif args.l2:
# 			col_prefix = "L2"; file_suffix = "l2.jknife"
# 			c = "Computing LD Score (L2) and block jackknife standard errors with {n} blocks."
# 			
# 		elif args.l4:
# 			col_prefix = "L4"; file_suffix = "l4.jknife"
# 			c = "Computing L4 and block jackknife standard errors with {n} blocks."
# 			
# 		print c.format(n=jN)
# 		(lN_est, lN_se) = geno_array.ldScoreBlockJackknife(block_left, args.chunk_size, jN=jN,
# 			annot=annot_matrix)
# 		lN = np.c_[lN_est, lN_se]
# 		if num_annots == 1:
# 			ldscore_colnames = [col_prefix+scale_suffix, 'SE('+col_prefix+scale_suffix+')']
# 		else:
# 			ldscore_colnames =  [x+col_prefix+scale_suffix for x in annot_colnames]
# 			ldscore_colnames += ['SE('+x+scale_suffix+')' for x in ldscore_colnames]

# 	else: # not block jackknife
# 		if args.l1:
# 			log.log("Estimating L1.")
# 			lN = geno_array.l1VarBlocks(block_left, args.chunk_size, annot=annot_matrix)
# 			col_prefix = "L1"; file_suffix = "l1"
# 		
# 		elif args.l1sq:
# 			log.log("Estimating L1 ^ 2.")
# 			lN = geno_array.l1sqVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
# 			col_prefix = "L1SQ"; file_suffix = "l1sq"
# 		
# 		elif args.l2:
# 			log.log("Estimating LD Score (L2).")
# 			lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
# 			col_prefix = "L2"; file_suffix = "l2"
# 				
# 		elif args.l4:
# 			col_prefix = "L4"; file_suffix = "l4"
# 			raise NotImplementedError('Sorry, havent implemented L4 yet. Try the jackknife.')
# 			lN = geno_array.l4VarBlocks(block_left, c, annot)
		
	log.log("Estimating LD Score.")
	lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
	col_prefix = "L2"; file_suffix = "l2"

	if num_annots == 1:
		ldscore_colnames = [col_prefix+scale_suffix]
	else:
		ldscore_colnames =  [x+col_prefix+scale_suffix for x in annot_colnames]
			
	# print .ldscore
	# output columns: CHR, BP, CM, RS, MAF, [LD Scores and optionally SEs]
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
		out_fname_pickle = out_fname+l2_suffix
		df.to_pickle(out_fname_pickle)
		
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
			annot_df.to_pickle(out_fname_annot_pickle)
			
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
	if num_annots > 1: # condition number of a column vector w/ nonzero var is trivially one
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


def sumstats(args, header=None):
	'''
	Wrapper function for estmating
		1. h2 / partitioned h2
		2. genetic covariance / correlation
		3. LD Score regression intercept
	
	from reference panel LD and GWAS summary statistics.
	
	'''
	
	# open output files
	log = logger(args.out + ".log")
	if header:
		log.log(header)
	
	# read .chisq or betaprod
	try:
		if args.h2:
			chisq = args.h2+'.chisq.gz'
			log.log('Reading summary statistics from {S}.'.format(S=chisq))
			sumstats = ps.chisq(chisq)
		elif args.intercept:
			chisq = args.intercept+'.chisq.gz'
			log.log('Reading summary statistics from {S}.'.format(S=chisq))
			sumstats = ps.chisq(chisq)
		elif args.rg:
			try:
				(p1, p2) = args.rg.split(',')
			except ValueError as e:
				log.log('Error: argument to --rg must be two .chisq/.allele fileset prefixes separated by a comma.')
				raise e
				
			chisq1 = p1 + '.chisq.gz'
			chisq2 = p2 + '.chisq.gz'
			allele1 = p1 + '.allele.gz'
			allele2 = p2 + '.allele.gz'
			sumstats = ps.betaprod_fromchisq(chisq1, chisq2, allele1, allele2)
	except ValueError as e:
		log.log('Error parsing summary statistics.')
		raise e
	
	log_msg = 'Read summary statistics for {N} SNPs.'
	log.log(log_msg.format(N=len(sumstats)))
	
	log.log('Reading LD Scores...')
	# read reference panel LD Scores
	try:
		if args.ref_ld:
			ref_ldscores = ps.ldscore(args.ref_ld)
		elif args.ref_ld_chr:
			ref_ldscores = ps.ldscore(args.ref_ld_chr, 22)
		elif args.ref_ld_file:
			ref_ldscores = ps.ldscore_fromfile(args.ref_ld_file)
		elif args.ref_ld_file_chr:
			ref_ldscores = ps.ldscore_fromfile(args.ref_ld_file_chr, 22)	
		elif args.ref_ld_list:
			flist = args.ref_ld_list.split(',')
			ref_ldscores = ps.ldscore_fromlist(flist)
		elif args.ref_ld_list_chr:
			flist = args.ref_ld_list_chr.split(',')
			ref_ldscores = ps.ldscore_fromlist(flist, 22)
		
	except ValueError as e:
		log.log('Error parsing reference LD.')
		raise e
				
	# read --M
	if args.M:
		try:
			M_annot = [float(x) for x in args.M.split(',')]
		except TypeError as e:
			raise TypeError('Count not case --M to float: ' + str(e.args))
		
		if len(M_annot) != len(ref_ldscores.columns) - 1:
			msg = 'Number of comma-separated terms in --M must match the number of partitioned'
			msg += 'LD Scores in --ref-ld'
			raise ValueError(msg)
		
	# read .M or --M-file			
	else:
		if args.M_file:
			if args.ref_ld:
				M_annot = ps.M(args.M_file)	
			elif args.ref_ld_chr:
				M_annot = ps.M(args.M_file, 22)
		elif args.not_M_5_50:
			if args.ref_ld:
				M_annot = ps.M(args.ref_ld)	
			elif args.ref_ld_chr:
				M_annot = ps.M(args.ref_ld_chr, 22)
		else:
			if args.ref_ld:
				M_annot = ps.M(args.ref_ld, common=True)	
			elif args.ref_ld_chr:
				M_annot = ps.M(args.ref_ld_chr, 22, common=True)
			elif args.ref_ld_file:
				M_annot = ps.M_fromfile(args.ref_ld_file)
			elif args.ref_ld_file_chr:
				M_annot = ps.M_fromfile(args.ref_ld_file_chr, 22)
			elif args.ref_ld_list:
				flist = args.ref_ld_list.split(',')
				M_annot = ps.M_fromlist(flist)
			elif args.ref_ld_list_chr:
				flist = args.ref_ld_list_chr.split(',')
				M_annot = ps.M_fromlist(flist, 22)
				
		# filter ref LD down to those columns specified by --keep-ld
		if args.keep_ld is not None:
			try:
				keep_M_indices = [int(x) for x in args.keep_ld.split(',')]
				keep_ld_colnums = [int(x)+1 for x in args.keep_ld.split(',')]
			except ValueError as e:
				raise ValueError('--keep-ld must be a comma-separated list of column numbers: '\
					+str(e.args))
	
			if len(keep_ld_colnums) == 0:
				raise ValueError('No reference LD columns retained by --keep-ld')
	
			keep_ld_colnums = [0] + keep_ld_colnums
			try:
				M_annot = [M_annot[i] for i in keep_M_indices]
				ref_ldscores = ref_ldscores.ix[:,keep_ld_colnums]
			except IndexError as e:
				raise IndexError('--keep-ld column numbers are out of bounds: '+str(e.args))
		
	log.log('Using M = '+str(np.array(M_annot)).replace('[','').replace(']','') ) # convert to np to use np printoptions
	ii = np.squeeze(np.array(ref_ldscores.iloc[:,1:len(ref_ldscores.columns)].var(axis=0) == 0))
	if np.any(ii):
		log.log('Removing partitioned LD Scores with zero variance')
		ii = np.insert(ii, 0, False) # keep the SNP column		
		ref_ldscores = ref_ldscores.ix[:,np.logical_not(ii)]
		M_annot = [M_annot[i] for i in xrange(1,len(ii)) if not ii[i]]
		n_annot = len(M_annot)
			
	log_msg = 'Read reference panel LD Scores for {N} SNPs.'
	log.log(log_msg.format(N=len(ref_ldscores)))

	# read regression SNP LD Scores
	try:
		if args.w_ld:
			w_ldscores = ps.ldscore(args.w_ld)
		elif args.w_ld_chr:
			w_ldscores = ps.ldscore(args.w_ld_chr, 22)

	except ValueError as e:
		log.log('Error parsing regression SNP LD')
		raise e
	
	if len(w_ldscores.columns) != 2:
		raise ValueError('--w-ld must point to a file with a single (non-partitioned) LD Score.')
	
	# to keep the column names from being the same
	w_ldscores.columns = ['SNP','LD_weights'] 

	log_msg = 'Read LD Scores for {N} SNPs to be retained for regression.'
	log.log(log_msg.format(N=len(w_ldscores)))
	
	# merge with reference panel LD Scores 
	sumstats = pd.merge(sumstats, ref_ldscores, how="inner", on="SNP")
	if len(sumstats) == 0:
		raise ValueError('No SNPs remain after merging with reference panel LD')
	else:
		log_msg = 'After merging with reference panel LD, {N} SNPs remain.'
		log.log(log_msg.format(N=len(sumstats)))

	# merge with regression SNP LD Scores
	sumstats = pd.merge(sumstats, w_ldscores, how="inner", on="SNP")
	if len(sumstats) <= 1:
		raise ValueError('No SNPs remain after merging with regression SNP LD')
	else:
		log_msg = 'After merging with regression SNP LD, {N} SNPs remain.'
		log.log(log_msg.format(N=len(sumstats)))
	
	ref_ld_colnames = ref_ldscores.columns[1:len(ref_ldscores.columns)]	
	w_ld_colname = sumstats.columns[-1]
	del(ref_ldscores); del(w_ldscores)
	
	err_msg = 'No SNPs retained for analysis after filtering on {C} {P} {F}.'
	log_msg = 'After filtering on {C} {P} {F}, {N} SNPs remain.'
	loop = ['1','2'] if args.rg else ['']
	var_to_arg = {'infomax': args.info_max, 'infomin': args.info_min, 'maf': args.maf}
	var_to_cname  = {'infomax': 'INFO', 'infomin': 'INFO', 'maf': 'MAF'}
	var_to_pred = {'infomax': lambda x: x < args.info_max, 
		'infomin': lambda x: x > args.info_min, 
		'maf': lambda x: x > args.maf}
	var_to_predstr = {'infomax': '<', 'infomin': '>', 'maf': '>'}
	for v in var_to_arg.keys():
		arg = var_to_arg[v]; pred = var_to_pred[v]; pred_str = var_to_predstr[v]
		for p in loop:
			cname = var_to_cname[v] + p; 
			if arg is not None:
				sumstats = ps.filter_df(sumstats, cname, pred)
				snp_count = len(sumstats)
				if snp_count == 0:
					raise ValueError(err_msg.format(C=cname, F=arg, P=pred_str))
				else:
					log.log(log_msg.format(C=cname, F=arg, N=snp_count, P=pred_str))

	# check condition number of LD Score Matrix
	if len(M_annot) > 1:
		cond_num = np.linalg.cond(sumstats[ref_ld_colnames])
		if cond_num > 100000:
			if args.invert_anyway:
				warn = "WARNING: LD Score matrix condition number is {C}. "
				warn += "Inverting anyway because the --invert-anyway flag is set."
				log.log(warn)
			else:
				warn = "WARNING: LD Score matrix condition number is {C}. "
				warn += "Remove collinear LD Scores or force inversion with "
				warn += "the --invert-anyway flag."
				log.log(warn.format(C=cond_num))
				raise ValueError(warn.format(C=cond_num))

	if len(sumstats) < 200000:
		log.log('WARNING: number of SNPs less than 200k; this is almost always bad.')

	# LD Score regression intercept
	if args.intercept:
		log.log('Estimating LD Score regression intercept.')
		# filter out large-effect loci
		max_N = np.max(sumstats['N'])
		if not args.no_filter_chisq:
			max_chisq = max(0.001*max_N, 20)
			sumstats = sumstats[sumstats['CHISQ'] < max_chisq]
			log_msg = 'After filtering on chi^2 < {C}, {N} SNPs remain.'
			log.log(log_msg.format(C=max_chisq, N=len(sumstats)))
	
			snp_count = len(sumstats)
			if snp_count == 0:
				raise ValueError(log_msg.format(C=max_chisq, N='no'))
			else:
				log.log(log_msg.format(C=max_chisq, N=len(sumstats)))

		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count

		log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))

		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		chisq = np.matrix(sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(sumstats.N).reshape((snp_count,1))
		del sumstats
		hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks)				
		log.log(hsqhat.summary_intercept())
		return hsqhat
		
	# LD Score regression to estimate h2
	elif args.h2:
	
		log.log('Estimating heritability.')
		max_N = np.max(sumstats['N'])
		if not args.no_filter_chisq:
			max_chisq = max(0.001*max_N, 80)
			sumstats = sumstats[sumstats['CHISQ'] < max_chisq]
			log_msg = 'After filtering on chi^2 < {C}, {N} SNPs remain.'
			log.log(log_msg.format(C=max_chisq, N=len(sumstats)))
			
		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count

		log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1,n_annot))
		chisq = np.matrix(sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(sumstats.N).reshape((snp_count,1))
		del sumstats

		if args.no_intercept:
			args.constrain_intercept = 1

		if args.constrain_intercept:
			try:
				intercept = float(args.constrain_intercept)
			except Exception as e:
				err_type = type(e).__name__
				e = ' '.join([str(x) for x in e.args])
				e = err_type+': '+e
				msg = 'Could not coerce argument to --constrain-intercept to floats.\n '+e
				raise ValueError(msg)
				
			log.log('Constraining LD Score regression intercept = {C}.'.format(C=intercept))
			hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks,
				args.non_negative, intercept)
					
		elif args.aggregate:
			if args.annot:
				annot = ps.AnnotFile(args.annot)
				num_annots,ma = len(annot.df.columns) - 4, len(annot.df)
				log.log("Read {A} annotations for {M} SNPs from {f}.".format(f=args.annot,
					A=num_annots,	M=ma))
				annot_matrix = np.matrix(annot.df.iloc[:,4:])
			else:
				raise ValueError("No annot file specified.")

			hsqhat = jk.Hsq_aggregate(chisq, ref_ld, w_ld, N, M_annot, annot_matrix, args.num_blocks)
		else:
			hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks, args.non_negative)
		
		if not args.human_only and n_annot > 1:
			hsq_cov_ofh = args.out+'.hsq.cov'
			_print_cov(hsqhat, hsq_cov_ofh, log)
					
		if args.print_delete_vals:
			hsq_delete_ofh = args.out+'.delete_k'
			_print_delete_k(hsqhat, hsq_delete_ofh, log)
	
		log.log(hsqhat.summary(ref_ld_colnames, args.overlap_annot))
		return [M_annot,hsqhat]


	# LD Score regression to estimate genetic correlation
	elif args.rg or args.rg or args.rg:
		log.log('Estimating genetic correlation.')

		max_N1 = np.max(sumstats['N1'])
		max_N2 = np.max(sumstats['N2'])
		if not args.no_filter_chisq:
			max_chisq1 = max(0.001*max_N1, 80)
			max_chisq2 = max(0.001*max_N2, 80)
			chisq1 = sumstats.BETAHAT1**2 * sumstats.N1
			chisq2 = sumstats.BETAHAT2**2 * sumstats.N2
			ii = np.logical_and(chisq1 < max_chisq1, chisq2 < max_chisq2)
			sumstats = sumstats[ii]
			log_msg = 'After filtering on chi^2 < ({C},{D}), {N} SNPs remain.'
			log.log(log_msg.format(C=max_chisq1, D=max_chisq2, N=np.sum(ii)))

		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count

		log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		betahat1 = np.matrix(sumstats.BETAHAT1).reshape((snp_count, 1))
		betahat2 = np.matrix(sumstats.BETAHAT2).reshape((snp_count, 1))
		N1 = np.matrix(sumstats.N1).reshape((snp_count,1))
		N2 = np.matrix(sumstats.N2).reshape((snp_count,1))
		del sumstats
		
		if args.no_intercept:
			args.constrain_intercept = "1,1,0"
		
		if args.constrain_intercept:
			intercepts = args.constrain_intercept.split(',')
			if len(intercepts) != 3:
				msg = 'If using --constrain-intercept with --sumstats-gencor, must specify a ' 
				msg += 'comma-separated list of three intercepts. '
				msg += 'The first two for the h2 estimates; the third for the gencov estimate.'
				raise ValueError(msg)
	
			try:
				intercepts = [float(x) for x in intercepts]
			except Exception as e:
				err_type = type(e).__name__
				e = ' '.join([str(x) for x in e.args])
				e = err_type+': '+e
				msg = 'Could not coerce arguments to --constrain-intercept to floats.\n '+e
				raise ValueError(msg)
			
			log.log('Constraining intercept for first h2 estimate to {I}'.format(I=str(intercepts[0])))
			log.log('Constraining intercept for second h2 estimate to {I}'.format(I=str(intercepts[1])))
			log.log('Constraining intercept for gencov estimate to {I}'.format(I=str(intercepts[2])))

		else:
			intercepts = [None, None, None]
		
		rghat = jk.Gencor(betahat1, betahat2, ref_ld, w_ld, N1, N2, M_annot, intercepts,
			args.overlap,	args.rho, args.num_blocks, return_silly_things=args.return_silly_things)

		if not args.human_only and n_annot > 1:
			gencov_jknife_ofh = args.out+'.gencov.cov'
			hsq1_jknife_ofh = args.out+'.hsq1.cov'
			hsq2_jknife_ofh = args.out+'.hsq2.cov'	
			_print_cov(rghat.hsq1, hsq1_jknife_ofh, log)
			_print_cov(rghat.hsq2, hsq2_jknife_ofh, log)
			_print_gencov_cov(rghat.gencov, gencov_jknife_ofh, log)
		
		if args.print_delete_vals:
			hsq1_delete_ofh = args.out+'.hsq1.delete_k'
			_print_delete_k(rghat.hsq1, hsq1_delete_ofh, log)
			hsq2_delete_ofh = args.out+'.hsq2.delete_k'
			_print_delete_k(rghat.hsq2, hsq2_delete_ofh, log)
			gencov_delete_ofh = args.out+'.gencov.delete_k'
			_print_delete_k(rghat.gencov, gencov_delete_ofh, log)

		log.log( '\n' )
		log.log( 'Heritability of first phenotype' )
		log.log( '-------------------------------' )
		log.log(rghat.hsq1.summary(ref_ld_colnames, args.overlap_annot))
		log.log( '\n' )
		log.log( 'Heritability of second phenotype' )
		log.log( '--------------------------------' )
		log.log(rghat.hsq2.summary(ref_ld_colnames, args.overlap_annot))
		log.log( '\n' )
		log.log( 'Genetic Covariance' )
		log.log( '------------------' )
		log.log(rghat.gencov.summary(ref_ld_colnames, args.overlap_annot))
		log.log( '\n' )
		log.log( 'Genetic Correlation' )
		log.log( '-------------------' )
		log.log(rghat.summary() )
		
		return [M_annot,rghat]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
		
	# Basic LD Score Estimation Flags
	parser.add_argument('--bfile', default=None, type=str, 
		help='Filename prefix for plink .bed/.bim/.fam file. '
		'The syntax is the same as with plink.  '
		'LDSC will automatically append .bed/.bim/.fam.')
	parser.add_argument('--l2', default=False, action='store_true',
		help='This flag tells LDSC to estimate LD Score. '
		'In order to estimate LD Score, you must also set the --bfile flag and '
		'one of the --ld-wind-* flags.')
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
	parser.add_argument('--cts-bin-add', default=None, type=str, 
		help='Same as --cts-bin, but tells LDSC to bin additively instead of multiplicatively. ')	
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

	

	# Basic Flags for Working with Variance Components
	parser.add_argument('--intercept', default=None, type=str,
		help='Filename prefix for a .chisq file for one-phenotype LD Score regression. '
		'--intercept performs the same analysis as --h2, but prints output '
		'focused on the LD Score regression intercept, rather than the h2 estimate. '
		'LDSC will automatically append .chisq or .chisq.gz to the filename prefix.'
		'--intercept requires at minimum also setting the --ref-ld and --w-ld flags.')
	parser.add_argument('--h2', default=None, type=str,
		help='Filename prefix for a .chisq file for one-phenotype LD Score regression. '
		'LDSC will automatically append .chisq or .chisq.gz to the filename prefix.'
		'--h2 requires at minimum also setting the --ref-ld and --w-ld flags.')
	parser.add_argument('--rg', default=None, type=str,
		help='Comma-separated list of two filename prefixes for .chisq/.allele filesets for '
		' two-phenotype LD Score regression. '
		'LDSC will automatically append .chisq/.allele or .chisq.gz/.allele.gz '
		' to the filename prefixes.'
		'--rg requires at minimum also setting the --ref-ld and --w-ld flags.')
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
	parser.add_argument('--ref-ld-file', default=None, type=str,
		help='File with one line per reference ldscore file, to be concatenated sideways.')
	parser.add_argument('--ref-ld-file-chr', default=None, type=str,
		help='Same as --ref-ld-file, but will concatenate LD Scores split into 22 '
		'chromosomes in the same manner as --ref-ld-chr.')
	parser.add_argument('--ref-ld-list', default=None, type=str,
		help='Comma-separated list of reference ldscore files, to be concatenated sideways.')
	parser.add_argument('--ref-ld-list-chr', default=None, type=str,
		help='Same as --ref-ld-list, except automatically concatenates LD Score files '
		'split into 22 chromosomes in the same manner as --ref-ld-chr.')

	parser.add_argument('--w-ld', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression.')
	parser.add_argument('--w-ld-chr', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression, split across 22 chromosomes.')
	
	parser.add_argument('--overlap-annot', default=False, action='store_true',
		help='This flag informs LDSC that the partitioned LD Scores were generates using an '
		'annot matrix with overlapping categories (i.e., not all row sums equal 1), '
		'and prevents LDSC from displaying output that is meaningless with overlapping categories.')

	parser.add_argument('--no-filter-chisq', default=False, action='store_true',
		help='Setting this flag prevents LDSC from removing huge-effect SNPs from the regression.')
	parser.add_argument('--no-intercept', action='store_true',
		help = 'If used with --h2, this constrains the LD Score regression intercept to equal '
		'1. If used with --rg, this constrains the LD Score regression intercepts for the h2 '
		'estimates to be one and the intercept for the genetic covariance estimate to be zero.')
	parser.add_argument('--constrain-intercept', action='store', default=False,
		help = 'If used with --h2, constrain the regression intercept to be a fixed value. '
		'If used with -rg, constrain the regression intercepts to a comma-separated list '
		'of three values, where the first value is the intercept of the first h2 regression, '
		'the second value is the intercept of the second h2 regression, and the third '
		'value is the intercept of the genetic covaraince regression (i.e., an estimate '
		'of (# of overlapping samples)*(phenotpyic correlation). ')
	parser.add_argument('--non-negative', action='store_true',
		help = 'Setting this flag causes LDSC to constrain all of the regression coefficients '
		'to be non-negative (i.e., to minimize the sum of squared errors subject to the '
		'constraint that all of the coefficients be positive. Note that the run-time is somewhat higher.')
	
	parser.add_argument('--M', default=None, type=str,	
		help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
	parser.add_argument('--M-file', default=None, type=str,
		help='Alternate .M file (e.g., if you want to use .M_5_50).')
		
	# Filtering for sumstats
	parser.add_argument('--info-min', default=None, type=float,
		help='Minimum INFO score for SNPs included in the regression. If your .chisq files '
		'do not include an INFO colum, setting this flag will result in an error. We '
		'recommend throwing out all low-INFO SNPs before making the .chisq file.')
	parser.add_argument('--info-max', default=None, type=float,
		help='Maximum INFO score for SNPs included in the regression. If your .chisq files '
		'do not include an INFO colum, setting this flag will result in an error.')
	parser.add_argument('--keep-ld', default=None, type=str,
		help='Zero-indexed column numbers of LD Scores to keep for LD Score regression.')
		
	# Optional flags for genetic correlation
	parser.add_argument('--overlap', default=0, type=int,
		help='By defualt LDSC weights the genetic covariance regression in --rg assuming that '
		'there are no overlapping samples. If there are overlapping samples, the LD Score '
		'regression standard error will be reduced if the weights take this into account. '
		'Use --overlap with --rg to tell LDSC the number of overlapping samples. '
		'--overlap must be used with --rho.  Since these numbers are only used for '
		'regression weights, it is OK if they are not precise.')
	parser.add_argument('--rho', default=0, type=float,
		help='Population correlation between phenotypes. Used only for weights in genetic covariance regression.')
	# Flags for both LD Score estimation and h2/gencor estimation
	parser.add_argument('--out', default='ldsc', type=str,
		help='Output filename prefix')
	parser.add_argument('--maf', default=None, type=float,
		help='Minor allele frequency lower bound. Default is 0')
	parser.add_argument('--human-only', default=False, action='store_true',
		help='Print only the human-readable .log file; do not print machine readable output.')
	# frequency (useful for .bin files)
	parser.add_argument('--print-delete-vals', default=False, action='store_true',
		help='Print block jackknife delete-k values.')


	# Flags you should almost never use
	parser.add_argument('--chunk-size', default=50, type=int,
		help='Chunk size for LD Score calculation. Use the default.')
	parser.add_argument('--pickle', default=False, action='store_true',
		help='Store .l2.ldscore files as pickles instead of gzipped tab-delimited text.')
	parser.add_argument('--yes-really', default=False, action='store_true',
		help='Yes, I really want to compute whole-chromosome LD Score')
	parser.add_argument('--aggregate', action='store_true',
		help = 'Use the aggregate estimator.')
	parser.add_argument('--invert-anyway', default=False, action='store_true',
		help="Force inversion of ill-conditioned matrices.")
	parser.add_argument('--num-blocks', default=200, type=int,
		help='Number of block jackknife blocks.')
	parser.add_argument('--not-M-5-50', default=False, action='store_true',
		help='Don\'t .M_5-50 file by default.')
	parser.add_argument('--return-silly-things', default=False, action='store_true',
		help='Force ldsc to return silly genetic correlation estimates.')

	# Out of commission flags	
	#parser.add_argument('--l1', default=False, action='store_true',
	#	help='Estimate l1 w.r.t. sample minor allele.')
	#parser.add_argument('--l1sq', default=False, action='store_true',
	#	help='Estimate l1 ^ 2 w.r.t. sample minor allele.')
	#parser.add_argument('--bin', default=None, type=str, 
	#	help='Prefix for binary VCF file')
	#parser.add_argument('--l4', default=False, action='store_true',
	#	help='Estimate l4. Only compatible with jackknife.')
	#parser.add_argument('--se', action='store_true', 
	#	help='Block jackknife SE? (Warning: somewhat slower)')
	#parser.add_argument('--freq', default=False, action='store_true',
	#	help='Compute reference allele frequencies (useful for .bin files).')

	args = parser.parse_args()
	defaults = vars(parser.parse_args(''))
	opts = vars(args)
	non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
	
	header = MASTHEAD
	header += "\nOptions: \n"
	options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
	header += '\n'.join(options).replace('True','').replace('False','')
	header += '\n'

	if args.w_ld:
		args.w_ld = args.w_ld
	elif args.w_ld_chr:
		args.w_ld_chr = args.w_ld_chr
	
	if args.num_blocks <= 1:
		raise ValueError('--num-blocks must be an integer > 1.')
	
	#if args.freq:	
	#	if (args.bfile is not None) == (args.bin is not None):
	#		raise ValueError('Must set exactly one of --bin or --bfile for use with --freq') 
	#
	#	freq(args, header)

	# LD Score estimation
	#elif (args.bin is not None or args.bfile is not None) and (args.l1 or args.l1sq or args.l2 or args.l4):
	#	if np.sum((args.l1, args.l2, args.l1sq, args.l4)) != 1:
	elif (args.bin is not None or args.bfile is not None):
		if args.l2 is None:
			#raise ValueError('Must specify exactly one of --l1, --l1sq, --l2, --l4 for LD estimation.')
			raise ValueError('Must specify --l2 with --bfile.')
		if args.bfile and args.bin:
			raise ValueError('Cannot specify both --bin and --bfile.')
		if args.annot is not None and args.extract is not None:
			raise ValueError('--annot and --extract are currently incompatible.')
		if args.cts_bin is not None and args.extract is not None:
			raise ValueError('--cts-bin and --extract are currently incompatible.')
		if args.annot is not None and args.cts_bin is not None:
			raise ValueError('--annot and --cts-bin are currently incompatible.')	
		if (args.cts_bin is not None or args.cts_bin_add is not None) != (args.cts_breaks is not None):
			raise ValueError('Must set both or neither of --cts-bin and --cts-breaks.')
		if args.per_allele and args.pq_exp is not None:
			raise ValueError('Cannot set both --per-allele and --pq-exp (--per-allele is equivalent to --pq-exp 1).')
		if args.per_allele:
			args.pq_exp = 1
		
		ldscore(args, header)
	
	# Summary statistics
	elif (args.h2 or 
		args.rg or 
		args.intercept or 
		args.rg) and\
		(args.ref_ld or args.ref_ld_chr or args.ref_ld_file or args.ref_ld_file_chr\
		 or args.ref_ld_list or args.ref_ld_list_chr) and\
		(args.w_ld or args.w_ld_chr):
		
		if np.sum(np.array((args.intercept, args.h2, args.rg)).astype(bool)) > 1:	
			raise ValueError('Cannot specify more than one of --h2, --rg, --intercept.')
		if args.ref_ld and args.ref_ld_chr:
			raise ValueError('Cannot specify both --ref-ld and --ref-ld-chr.')
		if args.ref_ld_list and args.ref_ld_list_chr:
			raise ValueError('Cannot specify both --ref-ld-list and --ref-ld-list-chr.')
		if args.ref_ld_file and args.ref_ld_file_chr:
			raise ValueError('Cannot specify both --ref-ld-list and --ref-ld-list-chr.')
		if args.w_ld and args.w_ld_chr:
			raise ValueError('Cannot specify both --w-ld and --w-ld-chr.')
		if args.rho or args.overlap:
			if not args.rg:
				raise ValueError('--rho and --overlap can only be used with --rg.')
			if not (args.rho and args.overlap):
				raise ValueError('Must specify either both or neither of --rho and --overlap.')
					
		sumstats(args, header)
		
		
	# bad flags
	else:
		print header
		print 'Error: no analysis selected.'
		print 'ldsc.py --help describes all options.'
