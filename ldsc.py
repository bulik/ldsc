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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=1000)

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
		
		
def _print_intercept(h2hat):
	'''
	Reusable code for printing information about LD Score regression intercept 
	from a jk.Hsq object
	
	'''
	
	out = []
	out.append( 'Observed scale h2: '+str(h2hat.tot_hsq)+' ('+str(h2hat.tot_hsq_se)+')')
	out.append( 'Lambda GC: '+ str(h2hat.lambda_gc))
	out.append( 'Mean Chi^2: '+ str(h2hat.mean_chisq))
	out.append( 'Weighted Mean Chi^2: '+ str(h2hat.w_mean_chisq))
	out.append( 'Intercept: '+ str(h2hat.intercept)+' ('+str(h2hat.intercept_se)+')')
	out.append( 'Ratio: '+str(h2hat.ratio)+' ('+str(h2hat.ratio_se)+')') 
	out = '\n'.join(out)
	return out
	
	
def _print_hsq(h2hat, ref_ld_colnames):
	'''Reusable code for printing output of jk.Hsq object'''
	out = []
	out.append('Total observed scale h2: '+str(h2hat.tot_hsq)+' ('+str(h2hat.tot_hsq_se)+')')
	out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
	out.append( 'Observed scale h2: '+str(h2hat.cat_hsq))
	out.append( 'Observed scale h2 SE: '+str(h2hat.cat_hsq_se))
	if h2hat.n_annot > 1:
		out.append( 'Proportion of SNPs: '+str(h2hat.M_prop))
		out.append( 'Proportion of h2g: ' +str(h2hat.prop_hsq))
		out.append( 'Enrichment: '+str(h2hat.enrichment))		
		
	out.append( 'Coefficients: '+str(h2hat.coef))
	out.append( 'Lambda GC: '+ str(h2hat.lambda_gc))
	out.append( 'Mean Chi^2: '+ str(h2hat.mean_chisq))
	out.append( 'Intercept: '+ str(h2hat.intercept)+' ('+str(h2hat.intercept_se)+')')
	out = '\n'.join(out)
	return out


def _print_hsq_nointercept(h2hat, ref_ld_colnames):
	'''Reusable code for printing output of jk.Hsq object'''
	out = []
	out.append('Total observed scale h2: '+str(h2hat.tot_hsq)+' ('+str(h2hat.tot_hsq_se)+')')
	out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
	out.append( 'Observed scale h2: '+str(h2hat.cat_hsq))
	out.append( 'Observed scale h2 SE: '+str(h2hat.cat_hsq_se))
	if h2hat.n_annot > 1:
		out.append( 'Proportion of SNPs: '+str(h2hat.M_prop))
		out.append( 'Proportion of h2g: ' +str(h2hat.prop_hsq))
		out.append( 'Enrichment: '+str(h2hat.enrichment))		
		
	out = '\n'.join(out)
	return out
	

def _print_gencov(gencov, ref_ld_colnames):
	'''Reusable code for printing output of jk.Gencov object'''
	out = []
	out.append('Total observed scale gencov: '+str(gencov.tot_gencov)+' ('+\
		str(gencov.tot_gencov_se)+')')
	out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
	out.append( 'Observed scale gencov: '+str(gencov.cat_gencov))
	out.append( 'Observed scale gencov SE: '+str(gencov.cat_gencov_se))
	if gencov.n_annot > 1:
		out.append( 'Proportion of SNPs: '+str(gencov.M_prop))
		out.append( 'Proportion of gencov: ' +str(gencov.prop_gencov))
		out.append( 'Enrichment: '+str(gencov.enrichment))		
		
	out.append( 'Intercept: '+ str(gencov.intercept)+' ('+str(gencov.intercept_se)+')')
	out = '\n'.join(out)
	return out

	
def _print_gencor(gencor):
	'''Reusable code for printing output of jk.Gencor object'''
	out = []
	out.append('Genetic Correlation: '+str(gencor.tot_gencor)+' ('+\
		str(gencor.tot_gencor_se)+')')
	out = '\n'.join(out)
	return out
	
	
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


def ldscore(args):
	'''
	Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
	reference panel genotypes. 
	
	Annot format is 
	chr snp bp cm <annotations>
	
	'''
	log = logger(args.out+'.log')
	
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
			for x in product(*full_labs)		:
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
	
	if args.se: # block jackknife

		# block size
		if args.block_size:
			jSize = args.block_size 
		elif n > 50:
			jSize = 10
		else:
			jSize = 1
		
		jN = int(np.ceil(n / jSize))
		if args.l1:
			col_prefix = "L1"; file_suffix = "l1.jknife"
			raise NotImplementedError('Sorry, havent implemented L1 block jackknife yet.')
			
		elif args.l1sq:
			col_prefix = "L1SQ"; file_suffix = "l1sq.jknife"
			raise NotImplementedError('Sorry, havent implemented L1^2 block jackknife yet.')
			
		elif args.l2:
			col_prefix = "L2"; file_suffix = "l2.jknife"
			c = "Computing LD Score (L2) and block jackknife standard errors with {n} blocks."
			
		elif args.l4:
			col_prefix = "L4"; file_suffix = "l4.jknife"
			c = "Computing L4 and block jackknife standard errors with {n} blocks."
			
		print c.format(n=jN)
		(lN_est, lN_se) = geno_array.ldScoreBlockJackknife(block_left, args.chunk_size, jN=jN,
			annot=annot_matrix)
		lN = np.c_[lN_est, lN_se]
		if num_annots == 1:
			ldscore_colnames = [col_prefix+scale_suffix, 'SE('+col_prefix+scale_suffix+')']
		else:
			ldscore_colnames =  [x+col_prefix+scale_suffix for x in annot_colnames]
			ldscore_colnames += ['SE('+x+scale_suffix+')' for x in ldscore_colnames]

	else: # not block jackknife
		if args.l1:
			log.log("Estimating L1.")
			lN = geno_array.l1VarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L1"; file_suffix = "l1"
		
		elif args.l1sq:
			log.log("Estimating L1 ^ 2.")
			lN = geno_array.l1sqVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L1SQ"; file_suffix = "l1sq"
		
		elif args.l2:
			log.log("Estimating LD Score (L2).")
			lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L2"; file_suffix = "l2"
				
		elif args.l4:
			col_prefix = "L4"; file_suffix = "l4"
			raise NotImplementedError('Sorry, havent implemented L4 yet. Try the jackknife.')
			lN = geno_array.l4VarBlocks(block_left, c, annot)
		
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
	
	log.log("Writing LD Scores for {N} SNPs to {f}.gz".format(f=out_fname, N=len(df)))
	df.to_csv(out_fname, sep="\t", header=True, index=False)	
	call(['gzip', '-f', out_fname])

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

	# print LD Score summary	
	pd.set_option('display.max_rows', 200)
	log.log('')
	log.log('Summary of {F}:'.format(F=out_fname))
	t = df.ix[:,4:].describe()
	log.log( t.ix[1:,:] )
	
	# print correlation matrix including all LD Scores and sample MAF
	log.log('')
	log.log('MAF/LD Correlation Matrix')
	log.log( df.ix[:,4:].corr() )
		

def sumstats(args):
	'''
	Wrapper function for estmating
		1. h2 / partitioned h2
		2. genetic covariance / correlation
		3. LD Score regression intercept
	
	from reference panel LD and GWAS summary statistics.
	
	'''
	
	# open output files
	log = logger(args.out + ".log")
	log.log(args)
	# read .chisq or betaprod
	try:
		if args.sumstats_h2:
			log.log('Reading summary statistics from {S}.'.format(S=args.sumstats_h2))
			sumstats = ps.chisq(args.sumstats_h2)
		elif args.sumstats_intercept:
			log.log('Reading summary statistics from {S}.'.format(S=args.sumstats_intercept))
			sumstats = ps.chisq(args.sumstats_intercept)
		elif args.sumstats_gencor:
			log.log('Reading summary statistics from {S}.'.format(S=args.sumstats_gencor))
			sumstats = ps.betaprod(args.sumstats_gencor)
		elif args.sumstats_gencor_fromchisq:
			if args.chisq1 and args.chisq2 and args.allele1 and args.allele2:
				sumstats = ps.betaprod_fromchisq(args.chisq1, args.chisq2, args.allele1, 
					args.allele2)
			elif args.chisq1 and args.chisq2:
				c1 = args.chisq1 + '.chisq.gz'
				c2 = args.chisq2 + '.chisq.gz'
				a1 = args.chisq1 + '.allele.gz'
				a2 = args.chisq2 + '.allele.gz'
				sumstats = ps.betaprod_fromchisq(c1, c2, a1, a2)
			else:
				raise ValueError('Must use --chisq1 and chisq2 flags with --sumstats-gencor-fromchisq.')
		elif args.gencor:
			(p1, p2) = args.gencor.split(',')
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
	
	# read reference panel LD Scores
	try:
		if args.ref_ld:
			ref_ldscores = ps.ldscore(args.ref_ld)
		elif args.ref_ld_chr:
			ref_ldscores = ps.ldscore(args.ref_ld_chr,22)

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
				
		# filter ref LD down to those columns specified by --keep-ld
		if args.keep_ld is not None:
			try:
				keep_M_indices = [int(x) for x in args.keep_ld.split(',')]
				keep_ld_colnums = [int(x)+1 for x in args.keep_ld.split(',')]
			except ValueError as e:
				raise ValueError('--keep-ld must be a comma-separate list of column numbers: '\
					+str(e.args))
	
			if len(keep_ld_colnums) == 0:
				raise ValueError('No reference LD columns retained by --keep-ld')
	
			keep_ld_colnums = [0] + keep_ld_colnums
			try:
				M_annot = [M_annot[i] for i in keep_M_indices]
				ref_ldscores = ref_ldscores.ix[:,keep_ld_colnums]
			except IndexError as e:
				raise IndexError('--keep-ld column numbers are out of bounds: '+str(e.args))
		
	log.log('Using M = '+str(M_annot))
	ii = ref_ldscores.iloc[:,1:len(ref_ldscores.columns)].var(axis=0) == 0
	if np.any(ii):
		log.log('Removing partitioned LD Scores with zero variance')
		ref_ldscores = ref_ldscores.ix[:,ii]
			
	log_msg = 'Read reference panel LD Scores for {N} SNPs.'
	log.log(log_msg.format(N=len(ref_ldscores)))

	# read regression SNP LD Scores
	try:
		if args.regression_snp_ld:
			w_ldscores = ps.ldscore(args.regression_snp_ld)
		elif args.regression_snp_ld_chr:
			w_ldscores = ps.ldscore(args.regression_snp_ld_chr, 22)

	except ValueError as e:
		log.log('Error parsing regression SNP LD')
		raise e
	
	# to keep the column names from being the same
	w_ldscores.columns = ['SNP','LD_weights'] 

	log_msg = 'Read LD Scores for {N} SNPs to be retained for regression.'
	log.log(log_msg.format(N=len(w_ldscores)))
	
	# merge with reference panel LD Scores 
	sumstats = pd.merge(sumstats, ref_ldscores, how="inner", on="SNP")
	log_msg = 'After merging with reference panel LD, {N} SNPs remain.'
	log.log(log_msg.format(N=len(sumstats)))

	# merge with regression SNP LD Scores
	sumstats = pd.merge(sumstats, w_ldscores, how="inner", on="SNP")
	log_msg = 'After merging with regression SNP LD, {N} SNPs remain.'
	log.log(log_msg.format(N=len(sumstats)))
	
	ref_ld_colnames = ref_ldscores.columns[1:len(ref_ldscores.columns)]	
	w_ld_colname = sumstats.columns[-1]
	del(ref_ldscores); del(w_ldscores)
	
	err_msg = 'No SNPs retained for analysis after filtering on {C} {P} {F}.'
	log_msg = 'After filtering on {C} {P} {F}, {N} SNPs remain.'
	loop = ['1','2'] if args.sumstats_gencor else ['']
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

	log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))
	if len(sumstats) < 200000:
		log.log('Note, # of SNPs < 200k; this is often bad.')

	# LD Score regression intercept
	if args.sumstats_intercept:
		log.log('Estimating LD Score regression intercept.')
		# filter out large-effect loci
		max_N = np.max(sumstats['N'])
		if not args.no_filter_chisq:
			max_chisq = max(0.001*max_N, 20)
			sumstats = sumstats[sumstats['CHISQ'] < max_chisq]
			log_msg = 'After filtering on chi^2 < {C}, {N} SNPs remain.'
	
		snp_count = len(sumstats)
		if snp_count == 0:
			raise ValueError(log_msg.format(C=max_chisq, N='no'))
		else:
			log.log(log_msg.format(C=max_chisq, N=len(sumstats)))

		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		chisq = np.matrix(sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(sumstats.N).reshape((snp_count,1))
		del sumstats
		h2hat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks)				
		log.log(_print_intercept(h2hat))
		return h2hat


	# LD Score regression to estimate h2
	elif args.sumstats_h2:
		log.log('Estimating heritability.')
		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		chisq = np.matrix(sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(sumstats.N).reshape((snp_count,1))
		del sumstats

		if args.no_intercept:
			log.log('Constraining LD Score regression intercept = 1.' )
			h2hat = jk.Hsq_nointercept(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks,
				args.non_negative)
			log.log(_print_hsq_nointercept(h2hat, ref_ld_colnames))
		elif args.aggregate:
			if args.annot:
				annot = ps.AnnotFile(args.annot)
				num_annots,ma = len(annot.df.columns) - 4, len(annot.df)
				log.log("Read {A} annotations for {M} SNPs from {f}.".format(f=args.annot,A=num_annots,
			M=ma))
				annot_matrix = np.matrix(annot.df.iloc[:,4:])
			else:
				raise ValueError("No annot file specified.")
			h2hat = jk.Hsq_aggregate(chisq, ref_ld, N, M_annot, annot_matrix, args.num_blocks)
			log.log(_print_hsq_nointercept(h2hat, ref_ld_colnames))
		else:
			h2hat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks,
				args.non_negative)
			log.log(_print_hsq(h2hat, ref_ld_colnames))
		return [M_annot,h2hat]
		
		if args.machine:
			hsq_jknife_ofh = args.out+'.hsq.jknife'
			np.savetxt(hsq_jknife_ofh, Hsq.hsq_cov)

	# LD Score regression to estimate genetic correlation
	elif args.sumstats_gencor or args.sumstats_gencor_fromchisq or args.gencor:
		log.log('Estimating genetic correlation.')
		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		betahat1 = np.matrix(sumstats.BETAHAT1).reshape((snp_count, 1))
		betahat2 = np.matrix(sumstats.BETAHAT2).reshape((snp_count, 1))
		N1 = np.matrix(sumstats.N1).reshape((snp_count,1))
		N2 = np.matrix(sumstats.N2).reshape((snp_count,1))
		del sumstats
		
		gchat = jk.Gencor(betahat1, betahat2, ref_ld, w_ld, N1, N2, M_annot, args.overlap,
			args.rho, args.num_blocks)

		log.log( '\n' )
		log.log( 'Heritability of first phenotype' )
		log.log( '-------------------------------' )
		log.log( _print_hsq(gchat.hsq1, ref_ld_colnames) )
		log.log( '\n' )
		log.log( 'Heritability of second phenotype' )
		log.log( '--------------------------------' )
		log.log( _print_hsq(gchat.hsq2, ref_ld_colnames) )
		log.log( '\n' )
		log.log( 'Genetic Covariance' )
		log.log( '------------------' )
		log.log( _print_gencov(gchat.gencov, ref_ld_colnames) )
		log.log( '\n' )
		log.log( 'Genetic Correlation' )
		log.log( '-------------------' )
		log.log( _print_gencor(gchat) )
		
		if args.machine:
			gencor_jknife_ofh = args.out+'.gencor.jknife'
			hsq1_jknife_ofh = args.out+'.hsq1.jknife'
			hsq2_jknife_ofh = args.out+'.hsq2.jknife'	
			np.savetxt(gencor_jknife_ofh, gchat.gencov.gencov_cov)
			np.savetxt(hsq1_jknife_ofh, gchat.hsq1.hsq_cov)
			np.savetxt(hsq2_jknife_ofh, gchat.hsq2.hsq_cov)

		return [M_annot,gchat]


def freq(args):
	'''
	Computes and prints reference allele frequencies. Identical to plink --freq. In fact,
	use plink --freq instead with .bed files; it's faster. This is useful for .bin files,
	which are a custom LDSC format.
	
	TODO: the MAF computation is inefficient, because it also filters the genotype matrix
	on MAF. It isn't so slow that it really matters, but fix this eventually. 
	
	'''
	log = logger(args.out+'.log')
	
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
	
	# read fam/ind
	array_indivs = ind_obj(ind_file)
	n = len(array_indivs.IDList)	 
	log.log('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))
	
	# read --extract
	if args.extract is not None:
		keep_snps = __filter__(args.extract, 'SNPs', 'include', array_snps)
	else:
		keep_snps = None
	
	# read keep_indivs
	if args.keep:
		keep_indivs = __filter__(args.keep, 'individuals', 'include', array_indivs)
	else:
		keep_indivs = None
	
	# read genotype array
	log.log('Reading genotypes from {fname}'.format(fname=array_file))
	geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
		keep_indivs=keep_indivs)
	
	frq_df = array_snps.df.ix[:,['CHR', 'SNP', 'A1', 'A2']]
	frq_array = np.zeros(len(frq_df))
	frq_array[geno_array.kept_snps] = geno_array.freq
	frq_df['FRQ'] = frq_array
	out_fname = args.out + '.frq'
	log.log('Writing reference allele frequencies to {O}.gz'.format(O=out_fname))
	frq_df.to_csv(out_fname, sep="\t", header=True, index=False)	
	call(['gzip', '-f', out_fname])


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
		help='Filename prefix for cts binned LD Score estimation')
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
		help='Chunk size for LD Score calculation. Use the default')

	# Output for LD Score
	parser.add_argument('--l1', default=False, action='store_true',
		help='Estimate l1 w.r.t. sample minor allele.')
	parser.add_argument('--l1sq', default=False, action='store_true',
		help='Estimate l1 ^ 2 w.r.t. sample minor allele.')
	parser.add_argument('--l2', default=False, action='store_true',
		help='Estimate l2. Compatible with both jackknife and non-jackknife.')
	parser.add_argument('--per-allele', default=False, action='store_true',
		help='Estimate per-allele l{N}. Same as --pq-exp 0. ')
	parser.add_argument('--pq-exp', default=None, type=float,
		help='Estimate l{N} with given scale factor. Default -1. Per-allele is equivalent to --pq-exp 1.')
	parser.add_argument('--l4', default=False, action='store_true',
		help='Estimate l4. Only compatible with jackknife.')
	parser.add_argument('--print-snps', default=None, type=str,
		help='Only print LD Scores for these SNPs.')
	parser.add_argument('--se', action='store_true', 
		help='Block jackknife SE? (Warning: somewhat slower)')
	parser.add_argument('--yes-really', default=False, action='store_true',
		help='Yes, I really want to compute whole-chromosome LD Score')

	# Summary Statistic Estimation Flags
	
	# Input for sumstats
	parser.add_argument('--sumstats-intercept', default=None, type=str,
		help='Path to file with summary statistics for LD Score regression estimation.')
	parser.add_argument('--sumstats-h2', default=None, type=str,
		help='Path to file with summary statistics for h2 estimation.')
	parser.add_argument('--sumstats-gencor', default=None, type=str,
		help='Path to file with summary statistics for genetic correlation estimation.')
	parser.add_argument('--sumstats-gencor-fromchisq',default=False,action='store_true',
		help='Make a betaprod dataframe from chisq1, chisq2, allele1, allele2.')
	parser.add_argument('--gencor',default=False,type=str)
	parser.add_argument('--chisq1',default=None,type=str,
		help='For use with --sumstats-gencor-fromchisq.')
	parser.add_argument('--chisq2',default=None,type=str,
		help='For use with --sumstats-gencor-fromchisq.')
	parser.add_argument('--allele1',default=None,type=str,
		help='For use with --sumstats-gencor-fromchisq.')
	parser.add_argument('--allele2',default=None,type=str,
		help='For use with --sumstats-gencor-fromchisq.')
	parser.add_argument('--intercept', default=False, action='store_true',
		help='For use with --sumstats-h2. Performs the same analysis as --sumstats-h2, but the output is focused on the LD Score regression intercept.')
	parser.add_argument('--ref-ld', default=None, type=str,
		help='Filename prefix for file with reference panel LD Scores.')
	parser.add_argument('--ref-ld-chr', default=None, type=str,
		help='Filename prefix for files with reference panel LD Scores split across 22 chromosomes.')
	parser.add_argument('--regression-snp-ld', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression.')
	parser.add_argument('--regression-snp-ld-chr', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression, split across 22 chromosomes.')
	parser.add_argument('--w-ld', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression.')
	parser.add_argument('--w-ld-chr', default=None, type=str,
		help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included in the regression, split across 22 chromosomes.')

	parser.add_argument('--no-filter-chisq', default=False, action='store_true',
		help='For use with --sumstats-intercept. Don\'t remove SNPs with large chi-square.')
	parser.add_argument('--no-intercept', action='store_true',
		help = 'Constrain the regression intercept to be 1.')
	parser.add_argument('--non-negative', action='store_true',
		help = 'Constrain the regression intercept to be 1.')
	parser.add_argument('--aggregate', action='store_true',
		help = 'Use the aggregate estimator.')
	parser.add_argument('--M', default=None, type=str,
		help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
	parser.add_argument('--M-file', default=None, type=str,
		help='Alternate .M file (e.g., if you want to use .M_5_50).')
	parser.add_argument('--M-5-50', default=False, action='store_true',
		help='Deprecated. Now default behavior.')
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
	parser.add_argument('--block-size', default=None, type=int, 
		help='Block size for block jackknife')
	parser.add_argument('--maf', default=None, type=float,
		help='Minor allele frequency lower bound. Default is 0')
	parser.add_argument('--machine', default=False, action='store_true',
		help='Enable machine-readable output?')
	# frequency (useful for .bin files)
	parser.add_argument('--freq', default=False, action='store_true',
		help='Compute reference allele frequencies (useful for .bin files).')
		
	args = parser.parse_args()
		
	if args.w_ld:
		args.regression_snp_ld = args.w_ld
	elif args.w_ld_chr:
		args.regression_snp_ld_chr = args.w_ld_chr
		
	if args.freq:
		if (args.bfile is not None) == (args.bin is not None):
			raise ValueError('Must set exactly one of --bin or --bfile for use with --freq') 
	
		freq(args)

	# LD Score estimation
	elif (args.bin is not None or args.bfile is not None) and (args.l1 or args.l1sq or args.l2 or args.l4):
		if np.sum((args.l1, args.l2, args.l1sq, args.l4)) != 1:
			raise ValueError('Must specify exactly one of --l1, --l1sq, --l2, --l4 for LD estimation.')
		if args.bfile and args.bin:
			raise ValueError('Cannot specify both --bin and --bfile.')
		if args.block_size is None: # default jackknife block size for LD Score regression
			args.block_size = 100
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
		
		ldscore(args)
	
	# Summary statistics
	elif (args.sumstats_h2 or 
		args.sumstats_gencor or 
		args.sumstats_intercept or 
		args.sumstats_gencor_fromchisq
		or args.gencor) and\
		(args.ref_ld or args.ref_ld_chr) and\
		(args.regression_snp_ld or args.regression_snp_ld_chr):
		
		if np.sum(np.array((args.sumstats_intercept, args.sumstats_h2, args.sumstats_gencor, args.gencor, args.sumstats_gencor_fromchisq)).astype(bool)) > 1:	
			raise ValueError('Cannot specify more than one of --sumstats-h2, --sumstats-gencor, --sumstats-intercept.')
		if args.ref_ld and args.ref_ld_chr:
			raise ValueError('Cannot specify both --ref-ld and --ref-ld-chr.')--pq-exp
		if args.regression_snp_ld and args.regression_snp_ld_chr:
			raise ValueError('Cannot specify both --regression-snp-ld and --regression-snp-ld-chr.')
		if args.rho or args.overlap:
			if not (args.sumstats_gencor or args.sumstats_gencor_fromchisq or args.gencor):
				raise ValueError('--rho and --overlap can only be used with --sumstats-gencor.')
			if not (args.rho and args.overlap):
				raise ValueError('Must specify either both or neither of --rho and --overlap.')
		
		if args.block_size is None: # default jackknife block size for h2/gencor
			args.block_size = 2000
			
		sumstats(args)
		
	# bad flags
	else:
		raise ValueError('No analysis selected.')