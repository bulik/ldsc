'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module contains objects implementing LD Score related analyses on GWAS summary 
statistics.

'''

from __future__ import division
import parse as ps
import numpy as np
import pandas as pd
import jackknife as jk
import parse as ps
import itertools as it

# complementary bases
COMPLEMENT = {
	'A': 'T',
	'T': 'A',
	'C': 'G',
	'G': 'C'
}

# bases
BASES = COMPLEMENT.keys()

# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]] 
	for x in it.product(BASES,BASES) 
	if x[0] != x[1]}

# True iff SNP 1 has the same alleles as SNP 2 (possibly w/ strand or ref allele flip)
MATCH_ALLELES = {''.join(x):
	((x[0] == x[2]) and (x[1] == x[3])) or # strand and ref match
	((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or # ref match, strand flip
	((x[0] == x[3]) and (x[1] == x[2])) or # ref flip, strand match
	((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]])) # strand and ref flip
	for x in it.product(BASES,BASES,BASES,BASES)
	if (x[0] != x[1]) and (x[2] != x[3]) and 
	not STRAND_AMBIGUOUS[''.join(x[0:2])] and
	not STRAND_AMBIGUOUS[''.join(x[2:4])]}

# True iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip (strand flip optional)
FLIP_ALLELES = {''.join(x):
	((x[0] == x[3]) and (x[1] == x[2])) or # strand match
	((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]])) # strand flip
	for x in it.product(BASES,BASES,BASES,BASES)
	if (x[0] != x[1]) and (x[2] != x[3]) and 
	(x[0] != COMPLEMENT[x[1]]) and (x[2] != COMPLEMENT[x[3]])
	and MATCH_ALLELES[''.join(x)]}

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


class _sumstats(object):
	'''
	Abstract base class implementing basic summary statistic functions. 
	
	One small compromise -- state is modified within methods only via log messages. 
	Waiting until the end of method and returning a log message is sometimes slow; this
	would be inconvenient for debugging and interactive users.
	
	'''

	def __init__(self, args, log, header=None):
		raise NotImplementedError
		
	def _read_ref_ld(self, args, log):
		'''Read reference LD Scores'''
		try:
			if args.ref_ld:
				log.log('Reading reference LD Scores from {F} ...'.format(F=args.ref_ld))
				ref_ldscores = ps.ldscore(args.ref_ld)
			elif args.ref_ld_chr:	
				if '@' in args.ref_ld_chr:
					f = args.ref_ld_chr.replace('@','[1-22]')
				else:
					f = args.ref_ld_chr+'[1-22]'
				log.log('Reading reference LD Scores from {F} ...'.format(F=f))
				ref_ldscores = ps.ldscore(args.ref_ld_chr, 22)
			elif args.ref_ld_file:
				log.log('Reading reference LD Scores listed in {F} ...'.format(F=args.ref_ld_file))
				ref_ldscores = ps.ldscore_fromfile(args.ref_ld_file)
			elif args.ref_ld_file_chr:
				log.log('Reading reference LD Scores listed in {F} ...'.format(F=args.ref_ld_file_chr))
				ref_ldscores = ps.ldscore_fromfile(args.ref_ld_file_chr, 22)		
			elif args.ref_ld_list:
				log.log('Reading list of reference LD Scores...')
				flist = args.ref_ld_list.split(',')
				ref_ldscores = ps.ldscore_fromlist(flist)	
			elif args.ref_ld_list_chr:
				log.log('Reading list of reference LD Scores...')
				flist = args.ref_ld_list_chr.split(',')
				ref_ldscores = ps.ldscore_fromlist(flist, 22)
					
		except ValueError as e:
			log.log('Error parsing reference LD.')
			raise e

		log_msg = 'Read reference panel LD Scores for {N} SNPs.'
		log.log(log_msg.format(N=len(ref_ldscores)))
		return ref_ldscores


	def _parse_sumstats(self, log, fh, require_alleles=False):
		try:
			chisq = fh+'.chisq.gz'
			self.log.log('Reading summary statistics from {S} ...'.format(S=chisq))
			sumstats = ps.chisq(chisq, require_alleles)
			
		except ValueError as e:
			self.log.log('Error parsing summary statistics.')
			raise e
		log_msg = 'Read summary statistics for {N} SNPs.'
		log.log(log_msg.format(N=len(sumstats)))
		
		return sumstats

	def _read_M(self, args, log):
		'''Read M (--M, --M-file, etc)'''
		if args.M:
			try:
				M_annot = [float(x) for x in args.M.split(',')]
			except TypeError as e:
				raise TypeError('Could not cast --M to float: ' + str(e.args))
		
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
			else:
				if args.ref_ld:
					M_annot = ps.M(args.ref_ld, common=True)	
				elif args.ref_ld_chr:
					M_annot = ps.M(args.ref_ld_chr, 22, common=True)
				elif args.ref_ld_file:
					M_annot = ps.M_fromfile(args.ref_ld_file, common=True)
				elif args.ref_ld_file_chr:
					M_annot = ps.M_fromfile(args.ref_ld_file_chr, 22, common=True)
				elif args.ref_ld_list:
					flist = args.ref_ld_list.split(',')
					M_annot = ps.M_fromlist(flist, common=True)
				elif args.ref_ld_list_chr:
					flist = args.ref_ld_list_chr.split(',')
					M_annot = ps.M_fromlist(flist, 22, common=True)
	
		return M_annot

	def _read_w_ld(self, args, log):
		'''Read regression SNP LD'''
		try:
			if args.w_ld:
				log.log('Reading regression weight LD Scores from {F} ...'.format(F=args.w_ld))
				w_ldscores = ps.ldscore(args.w_ld)
			elif args.w_ld_chr:
				if '@' in args.w_ld_chr:
					f = args.w_ld_chr.replace('@','[1-22]')
				else:
					f = args.w_ld_chr+'[1-22]'

				log.log('Reading regression weight LD Scores from {F} ...'.format(F=f))
				w_ldscores = ps.ldscore(args.w_ld_chr, 22)

		except ValueError as e:
			log.log('Error parsing regression SNP LD.')
			raise e
	
		if len(w_ldscores.columns) != 2:
			raise ValueError('--w-ld must point to a file with a single (non-partitioned) LD Score.')
	
		# to keep the column names from being the same
		w_ldscores.columns = ['SNP','LD_weights'] 

		log_msg = 'Read LD Scores for {N} SNPs to be retained for regression.'
		log.log(log_msg.format(N=len(w_ldscores)))
		return w_ldscores

	def _check_ld_condnum(self, args, log, M_annot, ref_ld):
		'''Check condition number of LD Score matrix'''
		if len(M_annot) > 1:
			cond_num = np.linalg.cond(ref_ld)
			if cond_num > 100000:
				cond_num = round(cond_num, 3)
				if args.invert_anyway:
					warn = "WARNING: LD Score matrix condition number is {C}. "
					warn += "Inverting anyway because the --invert-anyway flag is set."
					log.log(warn.format(C=cond_num))
				else:
					warn = "WARNING: LD Score matrix condition number is {C}. "
					warn += "Remove collinear LD Scores or force ldsc to use a pseudoinverse with "
					warn += "the --invert-anyway flag."
					log.log(warn.format(C=cond_num))
					raise ValueError(warn.format(C=cond_num))

	def _check_variance(self, log, M_annot, ref_ldscores):
		'''Remove zero-variance LD Scores'''
		ii = np.squeeze(np.array(ref_ldscores.iloc[:,1:len(ref_ldscores.columns)].var(axis=0) == 0))
		if np.all(ii):
			raise ValueError('All LD Scores have zero variance.')
		elif np.any(ii):
			log.log('Removing partitioned LD Scores with zero variance.')
			ii = np.insert(ii, 0, False) # keep the SNP column		
			ref_ldscores = ref_ldscores.ix[:,np.logical_not(ii)]
			M_annot = [M_annot[i] for i in xrange(1,len(ii)) if not ii[i]]
			n_annot = len(M_annot)

		return(M_annot, ref_ldscores)
		
	def _keep_ld(self, args, log, M_annot, ref_ldscores):
		'''Filter down to SNPs specified by --keep-ld'''
		if args.keep_ld is not None:
			try:
				keep_M_indices = [int(x) for x in args.keep_ld.split(',')]
				keep_ld_colnums = [int(x)+1 for x in args.keep_ld.split(',')]
			except ValueError as e:
				raise ValueError('--keep-ld must be a comma-separated list of column numbers: '\
					+str(e.args))

			if len(keep_ld_colnums) == 0:
				raise ValueError('No reference LD columns retained by --keep-ld.')

			keep_ld_colnums = [0] + keep_ld_colnums
			try:
				M_annot = [M_annot[i] for i in keep_M_indices]
				ref_ldscores = ref_ldscores.ix[:,keep_ld_colnums]
			except IndexError as e:
				raise IndexError('--keep-ld column numbers are out of bounds: '+str(e.args))
	
		log.log('Using M = '+jk.kill_brackets(str(np.array(M_annot))).replace(' ','') ) 
		return(M_annot, ref_ldscores)

	def _merge_sumstats_ld(self, args, log, sumstats, M_annot, ref_ldscores, w_ldscores):
		'''Merges summary statistics and LD into one data frame'''
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
		return(w_ld_colname, ref_ld_colnames, sumstats)
	
	def _filter(self, args, log, sumstats):
		raise NotImplementedError
		# TODO -- get this working again
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
	
	def _warn_length(self, log, sumstats):
		if len(sumstats) < 200000:
			log.log('WARNING: number of SNPs less than 200k; this is almost always bad.')

	def _print_cov(self, args, log, hsqhat, n_annot):
		'''Prints covariance matrix of slopes'''
		if not args.human_only and n_annot > 1:
			ofh = args.out+'.hsq_cov'
			log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
			np.savetxt(ofh, hsqhat.hsq_cov)

	def _print_gencov_cov(self, args, log, hsqhat):
		'''Prints covariance matrix of slopes'''
		if not args.human_only and n_annot > 1:
			hsq_cov_ofh = args.out+'.hsq.cov'
			self._print_cov(hsqhat, hsq_cov_ofh, self.log)
			log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
			np.savetxt(ofh, hsqhat.gencov_cov)

	def _print_delete_k(self, args, log, hsqhat):
		'''Prints block jackknife delete-k values'''
		if args.print_delete_vals:
			hsq_delete_ofh = args.out+'.delete_k'
			log.log('Printing block jackknife delete-k values to {F}'.format(F=ofh))
			out_mat = hsqhat._jknife.delete_values
			if hsqhat.constrain_intercept is None:
				ncol = out_mat.shape[1]
				out_mat = out_mat[:,0:ncol-1]
		
			np.savetxt(ofh, out_mat)


class H2(_sumstats):
	'''
	Implements h2 and partitioned h2 estimation
	'''
	def __init__(self, args, header):
		self.log = logger(args.out + ".log")
		if header:
			self.log.log(header)
		
		sumstats = self._parse_sumstats(self.log, args.h2)
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		w_ld_colname, ref_ld_colnames, self.sumstats =\
			self._merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		del sumstats
		self._check_ld_condnum(args, self.log, M_annot, self.sumstats[ref_ld_colnames])
		self._warn_length(self.log, self.sumstats)
		self.sumstats = self._filter_chisq(args, self.log, self.sumstats, 0.001)
		self.log.log('Estimating heritability.')
		snp_count = len(self.sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count
		
		log_msg = 'Estimating standard errors using a block jackknife with {N} blocks.'
		self.log.log(log_msg.format(N=args.num_blocks))
		ref_ld = np.matrix(self.sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(self.sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1,n_annot))
		chisq = np.matrix(self.sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(self.sumstats.N).reshape((snp_count,1))

		if args.no_intercept:
			args.constrain_intercept = 1

		if args.constrain_intercept:
			try:
				intercept = float(args.constrain_intercept)
			except Exception as e:
				err_type = type(e).__name__
				e = ' '.join([str(x) for x in e.args])
				e = err_type+': '+e
				msg = 'Could not cast argument to --constrain-intercept to float.\n '+e
				self.log.log('ValueError: '+msg)
				raise ValueError(msg)
				
			self.log.log('Constraining LD Score regression intercept = {C}.'.format(C=intercept))
			hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks,
				args.non_negative, intercept)
					
		elif args.aggregate:
			if args.annot:
				annot = ps.AnnotFile(args.annot)
				num_annots,ma = len(annot.df.columns) - 4, len(annot.df)
				self.log.log("Read {A} annotations for {M} SNPs from {f}.".format(f=args.annot,
					A=num_annots,	M=ma))
				annot_matrix = np.matrix(annot.df.iloc[:,4:])
			else:
				raise ValueError("No annot file specified.")

			hsqhat = jk.Hsq_aggregate(chisq, ref_ld, w_ld, N, M_annot, annot_matrix, args.num_blocks)
		else:
			hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks, args.non_negative)
				
		self._print_cov(args, self.log, hsqhat, n_annot)
		self._print_delete_k(args, self.log, hsqhat)	
				
	
		self.log.log(hsqhat.summary(ref_ld_colnames, args.overlap_annot))
		self.M_annot = M_annot
		self.hsqhat = hsqhat
		
	def _filter_chisq(self, args, log, sumstats, N_factor):
		max_N = np.max(sumstats['N'])
		if not args.no_filter_chisq:
			max_chisq = max(0.001*max_N, args.max_chisq)
			sumstats = sumstats[sumstats['CHISQ'] < max_chisq]
			log_msg = 'After filtering on chi^2 < {C}, {N} SNPs remain.'
			snp_count = len(sumstats)
			if snp_count == 0:
				raise ValueError(log_msg.format(C=max_chisq, N='no'))
			else:
				log.log(log_msg.format(C=max_chisq, N=snp_count))
				
		return sumstats
	

class Intercept(H2):
	'''
	LD Score regression intercept
	'''
	def __init__(self):
		self.log = logger(args.out + ".log")
		if header:
			self.log.log(header)
	
		sumstats = self._parse_sumstats(log, args.intercept)
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		self.sumstats = self_merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		self._check_ld_condnum(args, self.log, M_annot, sumstats[ref_ld_colnames])
		self._warn_length(self.log, sumstats)
		self.sumstats = self._filter_chisq(args, log, sumstats, 0.001)
		log.log('Estimating LD Score regression intercept.')
		# filter out large-effect loci
		snp_count = len(sumstats)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count

		log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))
		ref_ld = np.matrix(sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		chisq = np.matrix(sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(sumstats.N).reshape((snp_count,1))
		hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks)				
		log.log(hsqhat.summary_intercept())
		self.hsqhat = hsqhat


class Rg(_sumstats):
	'''
	Implements rg estimation with fixed LD Scores, one fixed phenotype, and a loop over
	a list (possibly with length one) of other phenotypes.
	
	'''
	def __init__(self, args, header):
		self.log = logger(args.out + ".log")
		if header:
			self.log.log(header)
	
		rg_file_list = args.rg.split(',')
		rg_suffix_list = [x.split('/')[-1] for x in rg_file_list]
		pheno1 = rg_file_list[0]
		out_prefix = args.out + rg_suffix_list[0]
		sumstats = self._parse_sumstats(self.log, pheno1, require_alleles=True)
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		self.sumstats = self_merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		
		self.M_annot = M_annot
		self.rghat = []
		
		for i,pheno2 in enumerate(rg_file_list[1:len(rg_file_list)]):
			sumstats2 = self._parse_sumstats(log, pheno2)
			out_prefix_loop = out_prefix + '_' + rg_suffix_list[i+1]
			sumstats_loop = self._merge_sumstats(self.sumstats, sumstats2, log)
			sumstats_loop = self._filter_chisq(sumstats_loop)
			rghat = self._rg(sumstats_loop, args, log)
			
			if not args.human_only and n_annot > 1:
				gencov_jknife_ofh = out_prefix_loop+'.gencov.cov'
				hsq1_jknife_ofh = out_prefix_loop+'.hsq1.cov'
				hsq2_jknife_ofh = out_prefix_loop+'.hsq2.cov'	
				_print_cov(rghat.hsq1, hsq1_jknife_ofh, log)
				_print_cov(rghat.hsq2, hsq2_jknife_ofh, log)
				_print_gencov_cov(rghat.gencov, gencov_jknife_ofh, log)
		
			if args.print_delete_vals:
				hsq1_delete_ofh = out_prefix_loop+'.hsq1.delete_k'
				_print_delete_k(rghat.hsq1, hsq1_delete_ofh, log)
				hsq2_delete_ofh = out_prefix_loop+'.hsq2.delete_k'
				_print_delete_k(rghat.hsq2, hsq2_delete_ofh, log)
				gencov_delete_ofh = out_prefix_loop+'.gencov.delete_k'
				_print_delete_k(rghat.gencov, gencov_delete_ofh, log)

			self.log.log( '\n' )
			self.log.log( 'Heritability of first phenotype' )
			self.log.log( '-------------------------------' )
			self.log.log(rghat.hsq1.summary(ref_ld_colnames, args.overlap_annot))
			self.log.log( '\n' )
			self.log.log( 'Heritability of second phenotype' )
			self.log.log( '--------------------------------' )
			self.log.log(rghat.hsq2.summary(ref_ld_colnames, args.overlap_annot))
			self.log.log( '\n' )
			self.log.log( 'Genetic Covariance' )
			self.log.log( '------------------' )
			self.log.log(rghat.gencov.summary(ref_ld_colnames, args.overlap_annot))
			self.log.log( '\n' )
			self.log.log( 'Genetic Correlation' )
			self.log.log( '-------------------' )
			self.log.log(rghat.summary() )
		
	
	def _merge_sumstats_sumstats(self, sumstats1, sumstats2):
		### TODO -- replace value error with warning message
		# better to throw a warning than die on an error if rg 50/100 doesn't work but the
		# other 99 are OK.
	
		# remove strand ambiguous SNPs
		strand1 = (sumstats1.INC_ALLELE+sumstats1.DEC_ALLELE).apply(lambda y: STRAND_AMBIGUOUS[y])
		strand2 = (sumstats2.INC_ALLELE+sumstats1.DEC_ALLELE).apply(lambda y: STRAND_AMBIGUOUS[y])
		sumstats1 = sumstats1[np.logical_not(strand1)]
		sumstats2 = sumstats1[np.logical_not(strand2)]
		
		# merge sumstats
		sumstats2.rename(columns={'INC_ALLELE': 'INC_ALLELE2'}, inplace=True)
		sumstats2.rename(columns={'DEC_ALLELE': 'DEC_ALLELE2'}, inplace=True)
		sumstats1['BETAHAT1'] = np.sqrt(sumstats1['CHISQ'])/np.sqrt(sumstats1['N'])
		sumstats2['BETAHAT2'] = np.sqrt(sumstats2['CHISQ'])/np.sqrt(sumstats2['N'])
		del sumstats1['CHISQ']
		del sumstats2['CHISQ']
		x = pd.merge(sumstats1, sumstats2, how='inner', on='SNP')
		if len(x) == 0:
			raise ValueError('No SNPs remain after merge.')

		# remove SNPs where the alleles do not match
		alleles = x.INC_ALLELE+x.DEC_ALLELE+x.INC_ALLELE2+x.DEC_ALLELE2
		match = alleles.apply(lambda y: MATCH_ALLELES[y])
		x = x[np.logical_not(nmatch)]
		if len(x) == 0:
			raise ValueError('All SNPs have mismatched alleles.')
			
		# flip sign of betahat where ref alleles differ
		alleles = x.INC_ALLELE+x.DEC_ALLELE+x.INC_ALLELE2+x.DEC_ALLELE2
		flip = alleles.apply(lambda y: FLIP_ALLELES[y])
		x['BETAHAT2'] *= flip
		del x['INC_ALLELE']
		del x['DEC_ALLELE']
		del x['INC_ALLELE2']
		del x['DEC_ALLELE2']
		return x
		
	def _rg(self, sumstats_loop, args, log):
		self._check_ld_condnum(args, self.log, M_annot, sumstats[ref_ld_colnames])
		self._warn_length(self.log, sumstats)
		self.sumstats = self._filter_chisq(args, log, sumstats, 0.001)

		self.log.log('Estimating genetic correlation.')
		snp_count = len(sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count

		self.log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=args.num_blocks))
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
				self.log.log('ValueError: '+msg)
				raise ValueError(msg)

			try:
				intercepts = [float(x) for x in intercepts]
			except Exception as e:
				err_type = type(e).__name__
				e = ' '.join([str(x) for x in e.args])
				e = err_type+': '+e
				msg = 'Could not coerce arguments to --constrain-intercept to floats.\n '+e
				self.log.log('ValueError: '+msg)
				raise ValueError(msg)
		
			self.log.log('Constraining intercept for first h2 estimate to {I}'.format(I=str(intercepts[0])))
			self.log.log('Constraining intercept for second h2 estimate to {I}'.format(I=str(intercepts[1])))
			self.log.log('Constraining intercept for gencov estimate to {I}'.format(I=str(intercepts[2])))

		else:
			intercepts = [None, None, None]
	
		rghat = jk.Gencor(betahat1, betahat2, ref_ld, w_ld, N1, N2, M_annot, intercepts,
			args.overlap,	args.rho, args.num_blocks, return_silly_things=args.return_silly_things)
		
		return rghat
			
	def _filter_chisq(self, args, log, sumstats, N_factor):
		max_N1 = np.max(sumstats['N1'])
		max_N2 = np.max(sumstats['N2'])
		if not args.no_filter_chisq:
			max_chisq1 = max(0.001*max_N1, args.max_chisq)
			max_chisq2 = max(0.001*max_N2, args.max_chisq)
			chisq1 = sumstats.BETAHAT1**2 * sumstats.N1
			chisq2 = sumstats.BETAHAT2**2 * sumstats.N2
			ii = np.logical_and(chisq1 < max_chisq1, chisq2 < max_chisq2)
			sumstats = sumstats[ii]
			log_msg = 'After filtering on chi^2 < ({C},{D}), {N} SNPs remain.'
			log.log(log_msg.format(C=max_chisq1, D=max_chisq2, N=np.sum(ii)))
	
		return sumstats
