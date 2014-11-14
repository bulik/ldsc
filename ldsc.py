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
import ldscore.sumstats as sumstats
import argparse
import numpy as np
import pandas as pd
from subprocess import call
from itertools import product

__version__ = '0.0.2 (alpha)'

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
				' order as the .bim or .snp file')
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
		df.set_index('SNP')
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
		
	# LD Score Estimation Flags
	
	# Input
	parser.add_argument('--bin', default=None, type=str, 
		help='Prefix for binary VCF file')
	parser.add_argument('--bfile', default=None, type=str, 
		help='Prefix for Plink .bed/.bim/.fam file')
	parser.add_argument('--annot', default=None, type=str, 
		help='Filename prefix for annotation file for partitioned LD Score estimation')
	parser.add_argument('--cts-bin', default=None, type=str, 
		help='Filenames for multiplicative cts binned LD Score estimation')
	parser.add_argument('--cts-bin-add', default=None, type=str, 
		help='Filenames for additive cts binned LD Score estimation')
	parser.add_argument('--cts-breaks', default=None, type=str, 
		help='Comma separated list of breaks for --cts-bin. Specify negative numbers with an N instead of a -')
	parser.add_argument('--cts-names', default=None, type=str, 
		help='Comma separated list of column names for --cts-bin.')

	# Filtering / Data Management for LD Score
	parser.add_argument('--extract', default=None, type=str, 
		help='File with SNPs to include in LD Score analysis, one ID per row.')
	parser.add_argument('--keep', default=None, type=str, 
		help='File with individuals to include in LD Score analysis, one ID per row.')
	parser.add_argument('--ld-wind-snps', default=None, type=int,
		help='LD Window in units of SNPs. Can only specify one --ld-wind-* option')
	parser.add_argument('--ld-wind-kb', default=None, type=float,
		help='LD Window in units of kb. Can only specify one --ld-wind-* option')
	parser.add_argument('--ld-wind-cm', default=None, type=float,
		help='LD Window in units of cM. Can only specify one --ld-wind-* option')
	parser.add_argument('--chunk-size', default=50, type=int,
		help='Chunk size for LD Score calculation. Use the default.')

	# Output for LD Score
	#parser.add_argument('--l1', default=False, action='store_true',
	#	help='Estimate l1 w.r.t. sample minor allele.')
	#parser.add_argument('--l1sq', default=False, action='store_true',
	#	help='Estimate l1 ^ 2 w.r.t. sample minor allele.')
	parser.add_argument('--l2', default=False, action='store_true',
		help='Estimate l2. Compatible with both jackknife and non-jackknife.')
	parser.add_argument('--per-allele', default=False, action='store_true',
		help='Estimate per-allele l{N}. Same as --pq-exp 0. ')
	parser.add_argument('--pq-exp', default=None, type=float,
		help='Estimate l{N} with given scale factor. Default -1. Per-allele is equivalent to --pq-exp 1.')
	parser.add_argument('--maf-exp', default=None, type=float,
		help='Estimate l{N} with given MAF scale factor.')
	#parser.add_argument('--l4', default=False, action='store_true',
	#	help='Estimate l4. Only compatible with jackknife.')
	parser.add_argument('--print-snps', default=None, type=str,
		help='Only print LD Scores for these SNPs.')
	#parser.add_argument('--se', action='store_true', 
	#	help='Block jackknife SE? (Warning: somewhat slower)')
	parser.add_argument('--yes-really', default=False, action='store_true',
		help='Yes, I really want to compute whole-chromosome LD Score')
	parser.add_argument('--no-print-annot', default=False, action='store_true',
		help='Do not print the annot matrix produced by --cts-bin.')
	parser.add_argument('--pickle', default=False, action='store_true',
		help='Store .l2.ldscore files as pickles instead of gzipped tab-delimited text.')

	# Summary Statistic Estimation Flags
	
	# Input for sumstats
	parser.add_argument('--intercept', default=None, type=str,
		help='Path to .chisq file with summary statistics for LD Score regression estimation.')
	parser.add_argument('--h2', default=None, type=str,
		help='Path prefix to .chisq file with summary statistics for h2 estimation.')
	parser.add_argument('--rg', default=None, type=str,
		help='Comma-separated list of prefixes of .chisq filed for genetic correlation estimation.')
	parser.add_argument('--rg-list', default=None, type=str,
		help='File containing a list of prefixes of .chisq files (one per line) for genetic correlation estimation.')
	parser.add_argument('--ref-ld', default=None, type=str,
		help='Filename prefix for file with reference panel LD Scores.')
	parser.add_argument('--ref-ld-chr', default=None, type=str,
		help='Filename prefix for files with reference panel LD Scores split across 22 chromosomes.')
	parser.add_argument('--ref-ld-file', default=None, type=str,
		help='File with one line per reference ldscore file, to be concatenated sideways.')
	parser.add_argument('--ref-ld-file-chr', default=None, type=str,
		help='File with one line per ref-ld-chr prefix, to be concatenated sideways.')
	parser.add_argument('--ref-ld-list', default=None, type=str,
		help='Comma-separated list of reference ldscore files, to be concatenated sideways.')
	parser.add_argument('--ref-ld-list-chr', default=None, type=str,
		help='Comma-separated list of ref-ld-chr prefix, to be concatenated sideways.')

	parser.add_argument('--w-ld', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression.')
	parser.add_argument('--w-ld-chr', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression, split across 22 chromosomes.')
	parser.add_argument('--overlap-annot', default=False, action='store_true',
		help='Let ldsc know that some categories overlap; adjust ouput accordingly.')
	parser.add_argument('--invert-anyway', default=False, action='store_true',
		help="Force inversion of ill-conditioned matrices.")
	parser.add_argument('--no-filter-chisq', default=False, action='store_true',
		help='Don\'t remove SNPs with large chi-square.')
	parser.add_argument('--max-chisq', default=None, type=float,
		help='Max chi^2 for SNPs in the regression.')

	parser.add_argument('--no-intercept', action='store_true',
		help = 'Constrain the regression intercept to be 1.')
	parser.add_argument('--constrain-intercept', action='store', default=False,
		help = 'Constrain the regression intercept to be a fixed value (or a comma-separated list of 3 values for rg estimation).')
	parser.add_argument('--non-negative', action='store_true',
		help = 'Constrain the slopes to be non-negative.')
	parser.add_argument('--aggregate', action='store_true',
		help = 'Use the aggregate estimator.')
	parser.add_argument('--M', default=None, type=str,
		help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
	parser.add_argument('--M-file', default=None, type=str,
		help='Alternate .M file (e.g., if you want to use .M_5_50).')
	parser.add_argument('--not-M-5-50', default=False, action='store_true',
		help='Don\'t .M_5-50 file by default.')
		
	# Filtering for sumstats
	parser.add_argument('--info-min', default=None, type=float,
		help='Minimum INFO score for SNPs included in the regression.')
	parser.add_argument('--info-max', default=None, type=float,
		help='Maximum INFO score for SNPs included in the regression.')
	parser.add_argument('--keep-ld', default=None, type=str,
		help='Zero-indexed column numbers of LD Scores to keep for LD Score regression.')
		
	# Optional flags for genetic correlation
	parser.add_argument('--overlap', default=0, type=int,
		help='Number of overlapping samples. Used only for weights in genetic covariance regression.')
	parser.add_argument('--rho', default=0, type=float,
		help='Population correlation between phenotypes. Used only for weights in genetic covariance regression.')
	parser.add_argument('--num-blocks', default=200, type=int,
		help='Number of block jackknife blocks.')
	# Flags for both LD Score estimation and h2/gencor estimation
	parser.add_argument('--out', default='ldsc', type=str,
		help='Output filename prefix')
	parser.add_argument('--maf', default=None, type=float,
		help='Minor allele frequency lower bound. Default is 0')
	parser.add_argument('--human-only', default=False, action='store_true',
		help='Print only the human-readable .log file; do not print machine readable output.')
	# frequency (useful for .bin files)
	parser.add_argument('--freq', default=False, action='store_true',
		help='Compute reference allele frequencies (useful for .bin files).')
	parser.add_argument('--print-delete-vals', default=False, action='store_true',
		help='Print block jackknife delete-k values.')
	parser.add_argument('--return-silly-things', default=False, action='store_true',
		help='Force ldsc to return silly genetic correlation estimates.')
	parser.add_argument('--no-check', default=True, action='store_false',
		help='Don\'t check the contents of chisq files. These checks can be slow, and are '
		'redundant for chisq files generated using sumstats_to_chisq.py.')
	parser.add_argument('--no-check-alleles', default=False, action='store_true',
		help='For rg estimation, skip checking whether the alleles match. This check is '
		'redundant for pairs of chisq files generated using sumstats_to_chisq.py with the '
		'--merge-alleles flag.')

	parser.add_argument('--print-coefficients',default=False,action='store_true',
		help='when categories are overlapping, print coefficients as well as heritabilities.')
	parser.add_argument('--frqfile', type=str, 
		help='For use with --overlap-annot. Provides allele frequencies to prune to common '
		'snps if --not-M-5-50 is not set.')


	args = parser.parse_args()
	
	if args.no_check_alleles:
		args.no_check = False

	defaults = vars(parser.parse_args(''))
	opts = vars(args)
	non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
	
	header = MASTHEAD
	header += "\nOptions: \n"
	options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
	header += '\n'.join(options).replace('True','').replace('False','')
	header += '\n'
	
	if args.constrain_intercept:
		args.constrain_intercept = args.constrain_intercept.replace('N','-')
	
	if args.w_ld:
		args.w_ld = args.w_ld
	elif args.w_ld_chr:
		args.w_ld_chr = args.w_ld_chr
	
	if args.num_blocks <= 1:
		raise ValueError('--num-blocks must be an integer > 1.')
	
	if args.freq:	
		if (args.bfile is not None) == (args.bin is not None):
			raise ValueError('Must set exactly one of --bin or --bfile for use with --freq') 
	
		freq(args, header)

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
		args.rg_list) and\
		(args.ref_ld or args.ref_ld_chr or args.ref_ld_file or args.ref_ld_file_chr\
		 or args.ref_ld_list or args.ref_ld_list_chr) and\
		(args.w_ld or args.w_ld_chr):
		
		if np.sum(np.array((args.intercept, args.h2, args.rg or args.rg_list)).astype(bool)) > 1:	
			raise ValueError('Cannot specify more than one of --h2, --rg, --intercept, --rg-list.')
		if args.ref_ld and args.ref_ld_chr:
			raise ValueError('Cannot specify both --ref-ld and --ref-ld-chr.')
		if args.ref_ld_list and args.ref_ld_list_chr:
			raise ValueError('Cannot specify both --ref-ld-list and --ref-ld-list-chr.')
		if args.ref_ld_file and args.ref_ld_file_chr:
			raise ValueError('Cannot specify both --ref-ld-list and --ref-ld-list-chr.')
		if args.w_ld and args.w_ld_chr:
			raise ValueError('Cannot specify both --w-ld and --w-ld-chr.')
		if args.rho or args.overlap:
			if not args.rg or args.rg_list:
				raise ValueError('--rho and --overlap can only be used with --rg.')
			if not (args.rho and args.overlap):
				raise ValueError('Must specify either both or neither of --rho and --overlap.')
		
		if args.rg or args.rg_list:
			sumstats.Rg(args, header)
		elif args.h2:
			sumstats.H2(args, header)
		elif args.intercept:
			sumstats.Intercept(args, header)		
		
	# bad flags
	else:
		print header
		print 'Error: no analysis selected.'
		print 'ldsc.py --help describes all options.'


# def freq(args):
# 	'''
# 	Computes and prints reference allele frequencies. Identical to plink --freq. In fact,
# 	use plink --freq instead with .bed files; it's faster. This is useful for .bin files,
# 	which are a custom LDSC format.
# 	
# 	TODO: the MAF computation is inefficient, because it also filters the genotype matrix
# 	on MAF. It isn't so slow that it really matters, but fix this eventually. 
# 	
# 	'''
# 	log = logger(args.out+'.log')
# 	if header:
# 		log.log(header)
# 		
# 	if args.bin:
# 		snp_file, snp_obj = args.bin+'.bim', ps.PlinkBIMFile
# 		ind_file, ind_obj = args.bin+'.ind', ps.VcfINDFile
# 		array_file, array_obj = args.bin+'.bin', ld.VcfBINFile
# 	elif args.bfile:
# 		snp_file, snp_obj = args.bfile+'.bim', ps.PlinkBIMFile
# 		ind_file, ind_obj = args.bfile+'.fam', ps.PlinkFAMFile
# 		array_file, array_obj = args.bfile+'.bed', ld.PlinkBEDFile
# 
# 	# read bim/snp
# 	array_snps = snp_obj(snp_file)
# 	m = len(array_snps.IDList)
# 	log.log('Read list of {m} SNPs from {f}'.format(m=m, f=snp_file))
# 	
# 	# read fam/ind
# 	array_indivs = ind_obj(ind_file)
# 	n = len(array_indivs.IDList)	 
# 	log.log('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))
# 	
# 	# read --extract
# 	if args.extract is not None:
# 		keep_snps = __filter__(args.extract, 'SNPs', 'include', array_snps)
# 	else:
# 		keep_snps = None
# 	
# 	# read keep_indivs
# 	if args.keep:
# 		keep_indivs = __filter__(args.keep, 'individuals', 'include', array_indivs)
# 	else:
# 		keep_indivs = None
# 	
# 	# read genotype array
# 	log.log('Reading genotypes from {fname}'.format(fname=array_file))
# 	geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
# 		keep_indivs=keep_indivs)
# 	
# 	frq_df = array_snps.df.ix[:,['CHR', 'SNP', 'A1', 'A2']]
# 	frq_array = np.zeros(len(frq_df))
# 	frq_array[geno_array.kept_snps] = geno_array.freq
# 	frq_df['FRQ'] = frq_array
# 	out_fname = args.out + '.frq'
# 	log.log('Writing reference allele frequencies to {O}.gz'.format(O=out_fname))
# 	frq_df.to_csv(out_fname, sep="\t", header=True, index=False)	
# 	call(['gzip', '-f', out_fname])
