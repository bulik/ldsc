'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module deals with getting all the data needed for LD Score regression from files
into memory and checking that the input makes sense. There is no math here. LD Score 
regression is implemented in the regressions module. 

'''
from __future__ import division
import numpy as np
import pandas as pd
import itertools as it
import scipy.stats as stats
import jackknife as jk
import parse as ps
import regressions as reg
import sys, traceback
_N_CHR = 22	
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = COMPLEMENT.keys()
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]] 
	for x in it.product(BASES,BASES) 
	if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x	for x in map(lambda y: ''.join(y), it.product(BASES,BASES))
	if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
# True iff SNP 1 has the same alleles as SNP 2 (possibly w/ strand or ref allele flip)
MATCH_ALLELES = {x for x in map(lambda y: ''.join(y), it.product(VALID_SNPS, VALID_SNPS))
	if ((x[0] == x[2]) and (x[1] == x[3])) or # strand and ref match
	((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or # ref match, strand flip
	((x[0] == x[3]) and (x[1] == x[2])) or # ref flip, strand match
	((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))} # strand and ref flip
# True iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip (strand flip optional)
FLIP_ALLELES = {''.join(x):
	((x[0] == x[3]) and (x[1] == x[2])) or # strand match
	((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]])) # strand flip
	for x in MATCH_ALLELES}
		
def _select_and_log(x, ii, log, msg):	
	'''Fiter down to rows that are True in ii. Log # of SNPs removed.'''
	old_len = len(x)
	new_len = ii.sum()
	if new_len == 0:
		raise ValueError(msg.format(N=0))
	else:
		x = x[ii]
		log.log(msg.format(N=new_len))
	return x

def smart_merge(x, y):
	'''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
	if len(x) == len(y) and (x.SNP == y.SNP).all():
		x = x.reset_index(drop=True)
		y = y.reset_index(drop=True).drop('SNP', 1)
		out = pd.concat([x, y], axis=1)
	else:
		out = pd.merge(x, y, how='inner', on='SNP')
	return out
		
def _read_ref_ld( args, log):
	'''Read reference LD Scores.'''
	ref_ld = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
		 'reference panel LD Score', ps.ldscore_fromlist)
	log.log('Read reference panel LD Scores for {N} SNPs.'.format(N=len(ref_ld)))
	return ref_ld

def _read_annot(args,log):
	'''Read annot matrix.'''	
	overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log, 
		'annot matrix', ps.annot, frqfile=args.frqfile)
	return overlap_matrix, M_tot

def _read_M(args, log, n_annot):
	'''Read M (--M, --M-file, etc).'''
	if args.M:
		try:
			M_annot = [float(x) for x in args.M.split(',')]
		except ValueError as e:
			raise ValueError('Could not cast --M to float: ' + str(e.args))
		if len(M_annot) != n_annot:
			raise ValueError('# terms in --M must match # of LD Scores in --ref-ld.')
		M_annot = np.array(M_annot).reshape((1, n_annot))
	else:
		if args.ref_ld:
			M_annot = ps.M_fromlist(args.ref_ld.split(','), common=(not args.not_M_5_50))	
		elif args.ref_ld_chr:
			M_annot = ps.M_fromlist(args.ref_ld_chr.split(','), _N_CHR, common=(not args.not_M_5_50))

	return M_annot

def _read_w_ld( args, log):
	'''Read regression SNP LD.'''
	if (args.w_ld and ',' in args.w_ld) or (args.w_ld_chr and ',' in args.w_ld_chr):
		raise ValueError('--w-ld must point to a single fileset (no commas allowed).')
	w_ld = _read_chr_split_files(args.w_ld_chr, args.w_ld, log, 
		'regression weight LD Score', ps.ldscore_fromlist)
	if len(w_ld.columns) != 2:
		raise ValueError('--w-ld may only have one LD Score column.')
	w_ld.columns = ['SNP','LD_weights'] # prevent colname conflicts w/ ref ld
	log.log('Read regression weight LD Scores for {N} SNPs.'.format(N=len(w_ld)))
	return w_ld

def _read_chr_split_files(chr_arg, not_chr_arg, log, noun, parsefunc, *kwargs):
	'''Read files split across 22 chromosomes (annot, ref_ld, w_ld).'''
	try:
		if not_chr_arg:
			log.log('Reading {N} from {F} ...'.format(F=not_chr_arg, N=noun))
			out = parsefunc(not_chr_arg.split(','))
		elif chr_arg:
			f = ps.sub_chr(chr_arg, '[1-22]')
			log.log('Reading {N} from {F} ...'.format(F=f, N=noun))
			out = parsefunc(chr_arg.split(','), _N_CHR)
	except ValueError as e:
		log.log('Error parsing {N}.'.format(N=noun))
		raise e
		
	return out

def _read_sumstats(args, log, fh, alleles=False, dropna=False):
	'''Parse summary statistics.'''
	log.log('Reading summary statistics from {S} ...'.format(S=fh))
	sumstats = ps.sumstats(fh, alleles=alleles, dropna=dropna)
	log_msg = 'Read summary statistics for {N} SNPs.'
	log.log(log_msg.format(N=len(sumstats)))
	m = len(sumstats)
	sumstats = sumstats.drop_duplicates(subset='SNP')
	if m > len(sumstats):
		log.log('Dropped {M} SNPs with duplicated rs numbers.'.format(M=m-len(sumstats)))
		
	return sumstats

def _check_ld_condnum(args, log, ref_ld):
	'''Check condition number of LD Score matrix.'''
	if len(ref_ld.shape) >= 2:
		cond_num = int(np.linalg.cond(ref_ld))
		if cond_num > 100000:
			if args.invert_anyway:
				warn = "WARNING: LD Score matrix condition number is {C}. "
				warn += "Inverting anyway because the --invert-anyway flag is set."
				log.log(warn.format(C=cond_num))
			else:
				warn = "WARNING: LD Score matrix condition number is {C}. "
				warn += "Remove collinear LD Scores. "
				raise ValueError(warn.format(C=cond_num))

def _check_variance(log, M_annot, ref_ld):
	'''Remove zero-variance LD Scores.'''
	### TODO is there a SNP column here?
	ii = ref_ld.var(axis=0) == 0
	if ii.all():
		raise ValueError('All LD Scores have zero variance.')
	elif ii.any():
		log.log('Removing partitioned LD Scores with zero variance.')
		ref_ld = ref_ld.ix[:,~ii]
		M_annot = M_annot[:,np.array(~ii)]

	return M_annot, ref_ld
		
def _warn_length(log, sumstats):
	if len(sumstats) < 200000:
		log.log('WARNING: number of SNPs less than 200k; this is almost always bad.')

def _print_cov(ldscore_reg, ofh, log):
	'''Prints covariance matrix of slopes.'''
	log.log('Printing covariance matrix of the estimates to {F}.'.format(F=ofh))
	np.savetxt(ofh, ldscore_reg.cat_cov)

def _print_delete_values(ldscore_reg, ofh, log):
	'''Prints block jackknife delete-k values'''
	log.log('Printing block jackknife delete values to {F}.'.format(F=ofh))
	np.savetxt(ofh, ldscore_reg.tot_delete_values)
	
def _overlap_output( args, overlap_matrix, M_annot, n_annot, hsqhat, category_names, M_tot):
		### TODO what is happening here???
		for i in range(n_annot):
			overlap_matrix[i,:] = overlap_matrix[i,:]/M_annot
		
		prop_hsq_overlap = np.dot(overlap_matrix,hsqhat.prop_hsq.T).reshape((1,n_annot))
		prop_hsq_overlap_var = np.diag(np.dot(np.dot(overlap_matrix,hsqhat.prop_hsq_cov),overlap_matrix.T))
		prop_hsq_overlap_se = np.sqrt(prop_hsq_overlap_var).reshape((1,n_annot))
		one_d_convert = lambda x : np.array(x)[0]
		prop_M_overlap = M_annot/M_tot
		enrichment = prop_hsq_overlap/prop_M_overlap
		enrichment_se = prop_hsq_overlap_se/prop_M_overlap
		enrichment_p = stats.chi2.sf(one_d_convert((enrichment-1)/enrichment_se)**2, 1)
		df = pd.DataFrame({
			'Category':category_names,
			'Prop._SNPs':one_d_convert(prop_M_overlap),
			'Prop._h2':one_d_convert(prop_hsq_overlap),
			'Prop._h2_std_error': one_d_convert(prop_hsq_overlap_se),
			'Enrichment': one_d_convert(enrichment),
			'Enrichment_std_error': one_d_convert(enrichment_se),
			'Enrichment_p': enrichment_p
			})
		df = df[['Category','Prop._SNPs','Prop._h2','Prop._h2_std_error','Enrichment','Enrichment_std_error','Enrichment_p']]
		if args.print_coefficients:
			df['Coefficient'] = one_d_convert(hsqhat.coef)
			df['Coefficient_std_error'] = hsqhat.coef_se
			df['Coefficient_z-score'] = one_d_convert(hsqhat.coef/hsqhat.coef_se)

		df = df[np.logical_not(df['Prop._SNPs'] > .9999)]
		df.to_csv(args.out+'.results',sep="\t",index=False)	

def _merge_and_log(ld, sumstats, noun, log):
	'''Wrap smart merge with log messages about # of SNPs.'''
	sumstats = smart_merge(ld, sumstats)
	msg = 'After merging with {F}, {N} SNPs remain.'
	if len(sumstats) == 0:
		raise ValueError(msg.format(N=len(sumstats), F=noun))
	else:
		log.log(msg.format(N=len(sumstats), F=noun))
	
	return sumstats

def _read_ld_sumstats(args, log, fh, alleles=False, dropna=True):
	sumstats = _read_sumstats(args, log, fh, alleles=alleles, dropna=dropna)
	ref_ld = _read_ref_ld(args, log)
	n_annot = len(ref_ld.columns) - 1
	M_annot = _read_M(args, log, n_annot)
	M_annot, ref_ld = _check_variance(log, M_annot, ref_ld)
	w_ld = _read_w_ld(args, log)
	sumstats = _merge_and_log(ref_ld, sumstats, 'reference panel LD', log)
	sumstats = _merge_and_log(sumstats, w_ld, 'regression SNP LD', log)
	w_ld_cname = sumstats.columns[-1]
	ref_ld_cnames = ref_ld.columns[1:len(ref_ld.columns)]	
	return M_annot, w_ld_cname, ref_ld_cnames, sumstats

def estimate_h2(args, log):
	'''Estimate h2 and partitioned h2.'''
	M_annot, w_ld_cname, ref_ld_cnames, sumstats = _read_ld_sumstats(args, log, args.h2)
	ref_ld = sumstats.as_matrix(columns=ref_ld_cnames)
	_check_ld_condnum(args, log, ref_ld_cnames)
	_warn_length(log, sumstats)
	n_snp = len(sumstats); n_annot = len(ref_ld_cnames)
	s = lambda x: np.array(x).reshape((n_snp, 1))
	n_blocks = min(n_snp, args.n_blocks)
	chisq = s(sumstats.Z**2)
	hsqhat = reg.Hsq(chisq, ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N), 
		M_annot, n_blocks=n_blocks, intercept=args.constrain_intercept, twostep=args.two_step)

	if args.print_cov:
		_print_cov(hsqhat, args.out+'.cov', log)
	if args.print_delete_vals:
		_print_delete_values(hsqhat, args.out+'.delete', log)	
	if args.overlap_annot:
		overlap_matrix, M_tot = _read_annot(args, log)
		_overlap_output(args, overlap_matrix, M_annot, n_annot, hsqhat, ref_ld_cnames, M_tot)

	log.log(hsqhat.summary(ref_ld_cnames, P=args.samp_prev, K=args.pop_prev)) # should have args.overlap_annot
	return hsqhat

def estimate_rg(args, log):
	'''Estimate rg between trait 1 and a list of other traits.'''
	rg_paths, rg_files = _parse_rg(args.rg)
	pheno1 = rg_paths[0]
	out_prefix = args.out + rg_files[0]
	M_annot, w_ld_cname, ref_ld_cnames, sumstats = _read_ld_sumstats(args, log, pheno1, 
		alleles=True, dropna=True)
	RG = []; n_annot = len(M_annot)
	for i, pheno2 in enumerate(rg_paths[1:len(rg_paths)]):
		log.log('Computing rg for phenotype {I}/{N}'.format(I=i+2, N=len(rg_paths)))	
		try:
			loop = _read_other_sumstats(args, log, pheno2, sumstats, ref_ld_cnames)
			rghat = _rg(loop, args, log, M_annot, ref_ld_cnames, w_ld_cname)	
			RG.append(rghat)
			_print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, i==0)
			out_prefix_loop = out_prefix + '_' + rg_files[i+1]
			if args.print_cov:
				_print_rg_cov(rghat, out_prefix_loop, log)		
			if args.print_delete_vals:
				_print_rg_delete_values(rghat, out_prefix_loop, log)
				
		except Exception as e: # keep going if phenotype 50/100 causes an error
			msg = 'ERROR computing rg for phenotype {I}/{N}, from file {F}.'
			log.log(msg.format(I=i+2, N=len(rg_paths), F=rg_paths[i+1]))
			ex_type, ex, tb = sys.exc_info()
			log.log( traceback.format_exc(ex)+'\n' )
			if len(RG) <= i: # if exception raised before appending to RG
				RG.append(None)
	
	log.log('\nSummary of Genetic Correlation Results\n'+_get_rg_table(rg_paths, RG, args))
	return RG

def _read_other_sumstats(args, log, pheno2, sumstats, ref_ld_cnames):
	loop = _read_sumstats(args, log, pheno2, alleles=True, dropna=False)
	loop = _merge_sumstats_sumstats(args, sumstats, loop, log) 
	loop = loop.dropna(how='any')
	alleles = loop.A1+loop.A2+loop.A1x+loop.A2x
	if not args.no_check_alleles:
		loop = _select_and_log(loop, _filter_alleles(alleles), log, 
			'{N} SNPs with valid alleles.')	
	
	loop['Z2'] = _align_alleles(loop.Z2, alleles)
	loop = loop.drop(['A1', 'A1x', 'A2', 'A2x'], axis=1)
	_check_ld_condnum(args, log, loop[ref_ld_cnames])
	_warn_length(log, loop)
	return loop

def _get_rg_table(rg_paths, RG, args):
	'''Print a table of genetic correlations.'''
	t = lambda attr: lambda obj: getattr(obj, attr, 'NA')
	x = pd.DataFrame()
	x['p1'] = [rg_paths[0] for i in xrange(1, len(rg_paths))]
	x['p2'] = rg_paths[1:len(rg_paths)]
	x['rg'] = map(t('rg_ratio') , RG)
	x['se'] = map(t('rg_se') , RG)
	x['p'] = map(t('p') , RG)
	if args.samp_prev is not None and args.pop_prev is not None and\
		all((i is not None for i in args.samp_prev)) and all((i is not None for it in args.pop_prev)):
		c = reg.h2_obs_to_liab(1, args.samp_prev[1], args.pop_prev[1])
		x['h2_liab'] = map(lambda x: c*x, map(t('tot'), map(t('hsq2') , RG)))
		x['h2_liab_se'] = map(lambda x: c*x, map(t('tot_se'), map(t('hsq2') , RG)))
	else:
		x['h2_obs'] = map(t('tot'), map(t('hsq2') , RG))
		x['h2_obs_se'] =  map(t('tot_se	'), map(t('hsq2') , RG))
	if args.constrain_intercept is None:
		x['h2_int'] = map(t('intercept'), map(t('hsq2') , RG))
		x['h2_int_se'] = map(t('intercept_se'), map(t('hsq2') , RG))
		x['gcov_int'] = map(t('intercept'), map(t('gencov') , RG))
		x['gcov_int_se'] = map(t('intercept_se'), map(t('gencov') , RG))

	return x.to_string(header=True, index=False)+'\n'
			
def _print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, print_hsq1):
	l = lambda x: x+''.join(['-' for i in range(len(x.replace('\n','')))])
	if args.samp_prev is None and args.pop_prev is None:
		args.samp_prev = [None, None]; args.pop_prev = [None, None]
	if print_hsq1:
		log.log(l('\nHeritability of phenotype 1\n'))
		log.log(rghat.hsq1.summary(ref_ld_cnames, P=args.samp_prev[0], K=args.pop_prev[0]))
	
	log.log(l('\nHeritability of phenotype {I}/{N}\n'.format(I=i+2, N=len(rg_paths))))
	log.log(rghat.hsq2.summary(ref_ld_cnames, P=args.samp_prev[1], K=args.pop_prev[1]))
	log.log(l('\nGenetic Covariance\n'))
	log.log(rghat.gencov.summary(ref_ld_cnames, P=args.samp_prev, K=args.pop_prev))
	log.log(l('\nGenetic Correlation\n'))
	log.log(rghat.summary()+'\n')

def _merge_sumstats_sumstats( args, sumstats1, sumstats2, log):
	'''Merge two sets of summary statistics.'''
	sumstats1.rename(columns={'N':'N1','Z':'Z1'}, inplace=True)			
	sumstats2.rename(columns={'A1':'A1x','A2':'A2x','N':'N2','Z':'Z2'}, inplace=True)			
	x = _merge_and_log(sumstats1, sumstats2, 'summary staistics', log)
	return x

def _filter_alleles(alleles):
	'''Remove bad variants (mismatched alleles, non-SNPs, strand ambiguous).'''
	ii = alleles.apply(lambda y: y in MATCH_ALLELES)
	return ii

def _align_alleles(z, alleles):
	'''Align Z1 and Z2 to same choice of ref allele (allowing for strand flip).'''
	z *= (-1)**alleles.apply(lambda y: FLIP_ALLELES[y])
	return z
	
def _rg(sumstats, args, log, M_annot, ref_ld_cnames, w_ld_cname):
	'''Run the regressions.'''
	n_snp = len(sumstats); n_annot = len(ref_ld_cnames)
	s = lambda x: np.array(x).reshape((n_snp, 1))
	n_blocks = min(args.n_blocks, n_snp)	
	ref_ld = sumstats.as_matrix(columns=ref_ld_cnames) # TODO is this the right shape?
	intercepts = [None, None, None]
	if args.constrain_intercept is not None:
		intercepts = args.constrain_intercept

	rghat = reg.RG(s(sumstats.Z1), s(sumstats.Z2), 
		ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N1), s(sumstats.N2), M_annot, 
		intercept_hsq1=intercepts[0], intercept_hsq2=intercepts[1], 
		intercept_gencov=intercepts[2], n_blocks=n_blocks, twostep=args.two_step) 
	
	return rghat

def _parse_rg(rg):
	'''Parse args.rg.'''
	rg_paths = rg.split(',')	
	rg_files = [x.split('/')[-1] for x in rg_paths]
	if len(rg_paths) < 2:
		raise ValueError('Must specify at least two phenotypes for rg estimation.')
	
	return rg_paths, rg_files

def _print_rg_delete_values(rg, fh, log):
	'''Print block jackknife delete values.'''
	_print_delete_values(rg.hsq1, fh+'.hsq1.delete', log)
	_print_delete_values(rg.hsq2, fh+'.hsq2.delete', log)
	_print_delete_values(rg.gencov, fh+'.gencov.delete', log)

def _print_rg_cov(rghat, fh, log):
	'''Print covariance matrix of estimates.'''		
	_print_cov(rghat.hsq1, fh+'.hsq1.cov', log)
	_print_cov(rghat.hsq2, fh+'.hsq2.cov', log)
	_print_cov(rghat.gencov, fh+'.gencov.cov', log)