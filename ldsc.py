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
			raise ValueError(f(error_msg))

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
	out.append( 'Intercept: '+ str(h2hat.intercept)+' ('+str(h2hat.intercept_se)+')')
	out.append( 'Ratio: '+str(h2hat.ratio))
	out = '\n'.join(out)
	return out
	
	
def _print_hsq(h2hat, ref_ld_colnames):
	'''Reusable code for printing output of jk.Hsq object'''
	out = []
	out.append('Total observed scale h2: '+str(h2hat.tot_hsq)+' ('+str(h2hat.tot_hsq_se)+')')
	out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
	out.append( 'Observed scale h2: '+str(h2hat.cat_hsq))
	out.append( 'Observed scale h2 SE: '+str(h2hat.cat_hsq_se))
	out.append( 'Proportion of SNPs: '+str(h2hat.M_prop))
	out.append( 'Proportion of h2g: ' +str(h2hat.prop_hsq))
	out.append( 'Enrichment: '+str(h2hat.enrichment))		
	out.append( 'Intercept: '+ str(h2hat.intercept)+' ('+str(h2hat.intercept_se)+')')
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
	

def ldscore(args):
	'''
	Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
	reference panel genotypes. 
	
	Annot format is 
	chr snp bp cm <annotations>
	
	'''
	log = logger(args.out+'.log')
	
	if args.bin:
		snp_file, snp_obj = args.bin+'.snp', ps.VcfSNPFile
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
	
	# read annot
	if args.annot and args.keep:
		raise ValueError('--annot and --keep are currently incompatible.')
	elif args.annot:
		annot = ps.AnnotFile(args.annot)
		num_annots,ma = len(annot.df.columns) - 4, len(annot.df)
		log.log("Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,A=num_annots,
			M=ma))
		annot_matrix = np.array(annot.df.iloc[:,4:])
		annot_colnames = annot.df.columns[4:]
		keep_snps = None
	elif args.keep:
		keep_snps = __filter__(args.keep, 'SNPs', 'include', array_snps)
		annot_matrix, annot_colnames, num_annots = None, None, 1
	else:
		annot_matrix, annot_colnames, keep_snps = None, None, None, 
		num_annots = 1
	
	# read fam/ind
	array_indivs = ind_obj(ind_file)
	n = len(array_indivs.IDList)	 
	log.log('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))
	# read keep_indivs
	if args.extract:
		keep_indivs = __filter__(args.extract, 'individuals', 'include', array_indivs)
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
			ldscore_colnames = [col_prefix, 'SE('+col_prefix+')']
		else:
			ldscore_colnames =  [x+col_prefix for x in annot_colnames]
			ldscore_colnames += ['SE('+x+')' for x in ldscore_colnames]

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
			ldscore_colnames = [col_prefix]
		else:
			ldscore_colnames =  [x+col_prefix for x in annot_colnames]
			
	# print .ldscore
	# output columns: CHR, BP, CM, RS, MAF, [LD Scores and optionally SEs]
	out_fname = args.out + '.' + file_suffix + '.ldscore'
	new_colnames = geno_array.colnames + ldscore_colnames
	df = pd.DataFrame(np.c_[geno_array.df, lN])
	df.columns = new_colnames
	log.log("Writing results to {f}".format(f=out_fname))
	df.to_csv(out_fname, sep="\t", header=True, index=False)	
	
	# print .M
	fout_M = open(args.out + '.'+ file_suffix +'.M','wb')
	if num_annots == 1:
		print >> fout_M, geno_array.m
	else:
		M = np.squeeze(np.sum(annot_matrix, axis=0))
		print >> fout_M, '\t'.join(map(str,M))

	fout_M.close()

	
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
			sumstats = ps.chisq(args.sumstats_h2)
		elif args.sumstats_intercept:
			sumstats = ps.chisq(args.sumstats_intercept)
		elif args.sumstats_gencor:
			sumstats = ps.betaprod(args.sumstats_gencor)
	except ValueError as e:
		log.log('Error parsing summary statistics.')
		raise e
	
	log_msg = 'Read summary statistics for {N} SNPs.'
	log.log(log_msg.format(N=len(sumstats)))
	
	# read reference panel LD Scores and .M 
	try:
		if args.ref_ld:
			ref_ldscores = ps.ldscore(args.ref_ld)
			M_annot = ps.M(args.ref_ld)
	
		elif args.ref_ld_chr:
			ref_ldscores = ps.ldscore(args.ref_ld_chr,22)
			M_annot = ps.M(args.ref_ld_chr, 22)
	except ValueError as e:
		log.log('Error parsing reference LD.')
		raise e
		
	if np.any(ref_ldscores.iloc[:,1:len(ref_ldscores.columns)].var(axis=0) == 0):
		raise ValueError('Zero-variance LD Score. Possibly an empty column?')

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
	
	# this has to be here, because pandas will modify duplicate column names on merge
	### TODO still not quite satisfactory -- what if user accidentally submits
	# the same file for ref_ld and w_ld? 
	# maybe don't merge, but just get the row numbers to prevent modification of colnames??
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

	# LD Score regression intercept
	if args.sumstats_intercept:
		log.log('Estimating LD Score regression intercept')
		# filter out large-effect loci
		max_N = np.max(sumstats['N'])
		max_chisq = max(0.01*max_N, 20)
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
		
		h2hat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks)
		log.log(_print_hsq(h2hat, ref_ld_colnames))
		return [M_annot,h2hat]

	# LD Score regression to estimate genetic correlation
	elif args.sumstats_gencor:
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

	# Filtering / Data Management for LD Score
	parser.add_argument('--extract', default=None, type=str, 
		help='File with individuals to include in LD Score analysis, one ID per row.')
	parser.add_argument('--keep', default=None, type=str, 
		help='File with SNPs to include in LD Score analysis, one ID per row.')
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
	parser.add_argument('--l4', default=False, action='store_true',
		help='Estimate l4. Only compatible with jackknife.')
	
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
	
	# Filtering for sumstats
	parser.add_argument('--info-min', default=None, type=float,
		help='Minimum INFO score for SNPs included in the regression.')
	parser.add_argument('--info-max', default=None, type=float,
		help='Maximum INFO score for SNPs included in the regression.')
		
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
	args = parser.parse_args()
	
	# LD Score estimation
	if (args.bin or args.bfile) and (args.l1 or args.l1sq or args.l2 or args.l4):
		if np.sum((args.l1, args.l2, args.l1sq, args.l4)) != 1:
			raise ValueError('Must specify exactly one of --l1, --l1sq, --l2, --l4 for LN estimation.')
		if args.bfile and args.bin:
			raise ValueError('Cannot specify both --bin and --bfile.')
		
		if args.block_size is None: # default jackknife block size for LD Score regression
			args.block_size = 100
		
		ldscore(args)
	
	# Summary statistics
	elif (args.sumstats_h2 or args.sumstats_gencor or args.sumstats_intercept) and\
		(args.ref_ld or args.ref_ld_chr) and\
		(args.regression_snp_ld or args.regression_snp_ld_chr):
	
		if np.sum(np.array((args.sumstats_intercept, args.sumstats_h2, args.sumstats_gencor)).astype(bool)) > 1:	
			raise ValueError('Cannot specify more than one of --sumstats-h2, --sumstats-gencor, --sumstats-intercept.')
		if args.ref_ld and args.ref_ld_chr:
			raise ValueError('Cannot specify both --ref-ld and --ref-ld-chr.')
		if args.regression_snp_ld and args.regression_snp_ld_chr:
			raise ValueError('Cannot specify both --regression-snp-ld and --regression-snp-ld-chr.')
		if args.rho or args.overlap:
			if not args.sumstats_gencor:
				raise ValueError('--rho and --overlap can only be used with --sumstats-gencor.')
			if not (args.rho and args.overlap):
				raise ValueError('Must specify either both or neither of --rho and --overlap')
		
		if args.block_size is None: # default jackknife block size for h2/gencor
			args.block_size = 2000
			
		sumstats(args)
	
	# bad flags
	else:
		raise ValueError('No analysis selected.')
