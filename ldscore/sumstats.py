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
import sys, traceback
import itertools as it
import time
from scipy import stats

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

VALID_SNPS = {''.join(x)
	for x in it.product(BASES,BASES) 
	if x[0] != x[1] and not STRAND_AMBIGUOUS[''.join(x)]}


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

def smart_merge(x, y):
	'''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
	if len(x.SNP) == len(y.SNP) and (x.SNP == y.SNP).all():
		x = x.reset_index(drop=True)
		y = y.reset_index(drop=True).drop('SNP',1)
		out = pd.concat([x, y], axis=1)
	
	else:
		out = pd.merge(x, y, how='inner', on='SNP')
	
	return out
	
class Logger(object):
	'''
	Lightweight logging.
	
	TODO: replace with logging module
	
	'''
	def __init__(self, fh):
		self.log_name = fh
		self.log_fh = open(fh, 'wb')
		
 	def log(self, msg):
		'''
		Print to log file and stdout with a single command.
		
		'''
		print >>self.log_fh, msg
		print msg


class _sumstats(object):
	'''
	Base class implementing basic summary statistic functions. 
	
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

	def _read_annot(self,args,log):
		'''Read annot matrix'''
		try:
			if args.ref_ld:
				log.log('Reading annot matrix from {F} ...'.format(F=args.ref_ld))
				[overlap_matrix, M_tot] = ps.annot([args.ref_ld],frqfile=args.frqfile)
			elif args.ref_ld_chr:	
				if '@' in args.ref_ld_chr:
					f = args.ref_ld_chr.replace('@','[1-22]')
				else:
					f = args.ref_ld_chr+'[1-22]'
				log.log('Reading annot matrices from {F} ...'.format(F=f))
				[overlap_matrix, M_tot] = ps.annot([args.ref_ld_chr], 22,frqfile=args.frqfile)
			elif args.ref_ld_file:
				log.log('Reading annot matrices listed in {F} ...'.format(F=args.ref_ld_file))
				[overlap_matrix, M_tot] = ps.annot_fromfile(args.ref_ld_file,frqfile=args.frqfile)
			elif args.ref_ld_file_chr:
				log.log('Reading annot matrices listed in {F} ...'.format(F=args.ref_ld_file_chr))
				[overlap_matrix, M_tot] = ps.annot_fromfile(args.ref_ld_file_chr, 22,frqfile=args.frqfile)		
			elif args.ref_ld_list:
				log.log('Reading annot matrices...')
				flist = args.ref_ld_list.split(',')
				[overlap_matrix, M_tot] = ps.annot(flist,frqfile=args.frqfile)	
			elif args.ref_ld_list_chr:
				log.log('Reading annot matrices...')
				flist = args.ref_ld_list_chr.split(',')
				[overlap_matrix, M_tot] = ps.annot(flist, 22,frqfile=args.frqfile)
					
		except ValueError as e:
			log.log('Error reading annot matrix.')
			raise e

		log_msg = 'Read annot matrix.'
		log.log(log_msg)

		return [overlap_matrix, M_tot]	

	def _parse_sumstats(self, args, log, fh, require_alleles=False, keep_na=False):
		# priority is pickle, bz2, gz, uncompressed
		chisq = fh + '.chisq'
		chisq += ps.which_compression(fh+'.chisq')[0]
		self.log.log('Reading summary statistics from {S} ...'.format(S=chisq))
		sumstats = ps.chisq(chisq, require_alleles, keep_na, args.no_check)
		log_msg = 'Read summary statistics for {N} SNPs.'
		log.log(log_msg.format(N=len(sumstats)))
		if args.no_check:
			m = len(sumstats)
			sumstats = sumstats.drop_duplicates(subset='SNP')
			if m > len(sumstats):
				log.log('Dropped {M} SNPs with duplicated rs numbers.'.format(M=m-len(sumstats)))
			
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
		ii = np.squeeze(np.array(ref_ldscores.iloc[:,0:len(ref_ldscores.columns)].var(axis=0) == 0))
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
	
		log.log('Using M = '+', '.join(map(str,np.array(M_annot))))
		#log.log('Using M = '+jk.kill_brackets(str(np.array(M_annot))).replace(' ','') ) 
		return(M_annot, ref_ldscores)

	def _merge_sumstats_ld(self, args, log, sumstats, M_annot, ref_ldscores, w_ldscores):
		'''Merges summary statistics and LD into one data frame'''
		sumstats = smart_merge(ref_ldscores, sumstats)
		if len(sumstats) == 0:
			raise ValueError('No SNPs remain after merging with reference panel LD')
		else:
			log_msg = 'After merging with reference panel LD, {N} SNPs remain.'
			log.log(log_msg.format(N=len(sumstats)))

		# merge with regression SNP LD Scores
		sumstats = smart_merge(sumstats, w_ldscores)
		if len(sumstats) <= 1:
			raise ValueError('No SNPs remain after merging with regression SNP LD')
		else:
			log_msg = 'After merging with regression SNP LD, {N} SNPs remain.'
			log.log(log_msg.format(N=len(sumstats)))

		w_ld_colname = sumstats.columns[-1]
		ref_ld_colnames = ref_ldscores.columns[1:len(ref_ldscores.columns)]	

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

	def _print_cov(self, args, log, hsqhat, n_annot, ofh=None):
		'''Prints covariance matrix of slopes'''
		if not args.human_only and n_annot > 1:
			if not ofh: ofh = args.out+'.hsq_cov'
			log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
			np.savetxt(ofh, hsqhat.cat_hsq_cov)

	def _print_gencov_cov(self, args, log, gencovhat, n_annot, ofh=None):
		'''Prints covariance matrix of slopes'''
		if not args.human_only and n_annot > 1:
			if not ofh: ofh = args.out+'.gencov_cov'
			log.log('Printing covariance matrix of the estimates to {F}'.format(F=ofh))
			np.savetxt(ofh, gencovhat.gencov_cov)

	def _print_delete_k(self, args, log, hsqhat):
		'''Prints block jackknife delete-k values'''
		if args.print_delete_vals:
			ofh = args.out+'.delete_k'
			log.log('Printing block jackknife delete-k values to {F}'.format(F=ofh))
			out_mat = hsqhat._jknife.delete_values
			if hsqhat.constrain_intercept is None:
				ncol = out_mat.shape[1]
				out_mat = out_mat[:,0:ncol-1]
		
			np.savetxt(ofh, out_mat)
	
	def _masthead_and_time(self, args, header):
		self.log = Logger(args.out + ".log")
		if header:
			self.log.log(header)
		
		self.log.log('Beginning analysis at {T}'.format(T=time.ctime()))
		self.start_time = time.time()
	
	def _print_end_time(self, args, log):
		log.log('Analysis finished at {T}'.format(T=time.ctime()) )
		time_elapsed = round(time.time()-self.start_time,2)
		log.log('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
	
	def _overlap_output(self, args, overlap_matrix, M_annot, n_annot, hsqhat, category_names, M_tot):

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

class H2(_sumstats):
	'''
	Implements h2 and partitioned h2 estimation
	'''
	def __init__(self, args, header):
		
		self._masthead_and_time(args, header)
		# WARNING: sumstats contains NA values to speed up merge

		if args.overlap_annot:
			[overlap_matrix, M_tot] = self._read_annot(args,self.log)
			
			#check that M_annot == np.sum(annot_matrix,axis=0) and n_annot == annot_matrix.shape[1]
			# make --overlap-annot versions of __keep_ld and _check_variance
			
		sumstats = self._parse_sumstats(args, self.log, args.h2, keep_na=True)
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		#update the next three functions
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		w_ld_colname, ref_ld_colnames, self.sumstats =\
			self._merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		del sumstats

		# Remove NA values from sumstats
		ii = self.sumstats.CHISQ.notnull()
		self.log.log('{N} SNPs with nonmissing values.'.format(N=ii.sum()))
		self.sumstats = self.sumstats[ii]
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
			hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks, args.non_negative,
				slow=args.slow)
				
		self._print_cov(args, self.log, hsqhat, n_annot)
		self._print_delete_k(args, self.log, hsqhat)	
		
		if args.overlap_annot:
			self._overlap_output(args, overlap_matrix, M_annot, n_annot, hsqhat, ref_ld_colnames, M_tot)

		self.log.log(hsqhat.summary(ref_ld_colnames, args.overlap_annot, args.out))
		self.M_annot = M_annot
		self.hsqhat = hsqhat
		self.log.log('\n')
		self._print_end_time(args, self.log)
		
	def _filter_chisq(self, args, log, sumstats, N_factor):
		max_N = np.max(sumstats['N'])
		if not args.no_filter_chisq:
			if args.max_chisq is None:
				max_chisq_min = 80
				max_chisq = max(N_factor*max_N, max_chisq_min)
			else:
				max_chisq = args.max_chisq
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
	def __init__(self, args, header):
		self._masthead_and_time(args, header)
		# WARNING: sumstats contains NA values to speed up merge
		sumstats = self._parse_sumstats(args, self.log, args.intercept, keep_na=True)
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		w_ld_colname, ref_ld_colnames, self.sumstats =\
			self._merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		del sumstats
		# Remove NA values from sumstats
		ii = self.sumstats.CHISQ.notnull()
		self.log.log('{N} SNPs with nonmissing values.'.format(N=ii.sum()))
		self.sumstats = self.sumstats[ii]
		self._check_ld_condnum(args, self.log, M_annot, self.sumstats[ref_ld_colnames])
		self._warn_length(self.log, self.sumstats)
		self.sumstats = self._filter_chisq(args, self.log, self.sumstats, 0.001)
		self.log.log('Estimating LD Score regression intercept.')
		# filter out large-effect loci
		snp_count = len(self.sumstats); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			args.num_blocks = snp_count
		
		log_msg = 'Estimating standard errors using a block jackknife with {N} blocks.'
		self.log.log(log_msg.format(N=args.num_blocks))
		ref_ld = np.matrix(self.sumstats[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(self.sumstats[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		chisq = np.matrix(self.sumstats.CHISQ).reshape((snp_count, 1))
		N = np.matrix(self.sumstats.N).reshape((snp_count,1))
		hsqhat = jk.Hsq(chisq, ref_ld, w_ld, N, M_annot, args.num_blocks, slow=args.slow)				
		self.log.log(hsqhat.summary_intercept())
		self.hsqhat = hsqhat
		self._print_end_time(args, self.log)


class Rg(_sumstats):
	'''
	Implements rg estimation with fixed LD Scores, one fixed phenotype, and a loop over
	a list (possibly with length one) of other phenotypes.
	
	'''
	
	def __init__(self, args, header):
		self._masthead_and_time(args, header)
		if args.rg:
			rg_file_list = args.rg.split(',')
		elif args.rg_list:
			rg_file_list = [x.rstrip('\n') for x in open(args.rg_list,'r').readlines()]
			self.log.log('The following files are specified by --rg-list:')
			self.log.log('\n'.join(rg_file_list))
			self.log.log('\n')
			
		rg_suffix_list = [x.split('/')[-1] for x in rg_file_list]
		if len(rg_file_list) < 2:
			raise ValueError('Must specify at least two phenotypes for rg estimation.')
		pheno1 = rg_file_list[0]
		out_prefix = args.out + rg_suffix_list[0]
		sumstats = self._parse_sumstats(args, self.log, pheno1, require_alleles=True, 
			keep_na=True)	
		ref_ldscores = self._read_ref_ld(args, self.log)
		M_annot = self._read_M(args, self.log)
		M_annot, ref_ldscores = self._keep_ld(args, self.log, M_annot, ref_ldscores)
		M_annot, ref_ldscores = self._check_variance(self.log, M_annot, ref_ldscores)
		w_ldscores = self._read_w_ld(args, self.log)
		w_ld_colname, ref_ld_colnames, self.sumstats =\
		 self._merge_sumstats_ld(args, self.log, sumstats, M_annot, ref_ldscores, w_ldscores)
		self.M_annot = M_annot
		self.rghat = []
		self.rghat_se = []
		self.Z = []
		self.P = []
		for i,pheno2 in enumerate(rg_file_list[1:len(rg_file_list)]):
			if len(rg_file_list) > 2:
				log_msg = 'Computing genetic correlation for phenotype {I}/{N}'
				self.log.log(log_msg.format(I=i+2, N=len(rg_file_list)))
			else:
				self.log.log('Computing genetic correlation.')
			try:
				sumstats2 = self._parse_sumstats(args, self.log, pheno2, require_alleles=True, 
					keep_na=True)
				out_prefix_loop = out_prefix + '_' + rg_suffix_list[i+1]
				sumstats_loop = self._merge_sumstats_sumstats(args, self.sumstats, sumstats2, self.log)
				# missing data has now been removed
				sumstats_loop = self._filter_chisq(args, self.log, sumstats_loop, 0.001)
				self._check_ld_condnum(args, self.log, M_annot, sumstats_loop[ref_ld_colnames])
				self._warn_length(self.log, sumstats_loop)
				snp_count = len(sumstats_loop); n_annot = len(ref_ld_colnames)
				#if i == 0:
				#	rghat = self._rg(sumstats_loop, args, self.log, M_annot, ref_ld_colnames, 
				#		w_ld_colname)
				#	hsq1 = rghat.hsq1
				#else:
				#	rghat = self._rg(sumstats_loop, args, self.log, M_annot, ref_ld_colnames, 
				#		w_ld_colname, first_hsq=hsq1)
				
				# estimate hsq1 each time so that the jackknife blocks use the same SNPs
				# this is clearly not optimal w.r.t. runtime
				rghat = self._rg(sumstats_loop, args, self.log, M_annot, ref_ld_colnames, 
					w_ld_colname)
				if not args.human_only and n_annot > 1:
					gencov_jknife_ofh = out_prefix_loop+'.gencov.cov'
					hsq1_jknife_ofh = out_prefix_loop+'.hsq1.cov'
					hsq2_jknife_ofh = out_prefix_loop+'.hsq2.cov'	
					self._print_cov(args, self.log, rghat.hsq1, n_annot, hsq1_jknife_ofh)
					self._print_cov(args, self.log, rghat.hsq2, n_annot, hsq2_jknife_ofh)
					self._print_gencov_cov(args, self.log, rghat.gencov, n_annot, gencov_jknife_ofh)
		
				if args.print_delete_vals:
					hsq1_delete_ofh = out_prefix_loop+'.hsq1.delete_k'
					self._print_delete_k(rghat.hsq1, hsq1_delete_ofh, self.log)
					hsq2_delete_ofh = out_prefix_loop+'.hsq2.delete_k'
					self._print_delete_k(rghat.hsq2, hsq2_delete_ofh, self.log)
					gencov_delete_ofh = out_prefix_loop+'.gencov.delete_k'
					self._print_delete_k(rghat.gencov, gencov_delete_ofh, self.log)
			
				self._print_gencor(args, self.log, rghat, ref_ld_colnames, i, rg_file_list, i==0)
				self.rghat.append(rghat.tot_gencor)
				self.rghat_se.append(rghat.tot_gencor_se)
				self.Z.append(rghat.Z)
				self.P.append(rghat.P_val)

			except Exception as e:
				'''
				Better to print an error message then keep going if phenotype 50/100 causes an
				error but the other 99/100 phenotypes are OK
				
				'''
				
				msg = 'ERROR computing rg for phenotype {I}/{N}, from file {F}.'
				self.log.log(msg.format(I=i+2, N=len(rg_file_list), F=rg_file_list[i+1]))
				ex_type, ex, tb = sys.exc_info()
				self.log.log( traceback.format_exc(ex) )
				self.log.log('\n')
				self.rghat_se.append('NA')
				self.rghat.append('NA')
				self.Z.append('NA')
				self.P.append('NA')
				
		self.log.log('Summary of Genetic Correlation Results')

		x = pd.DataFrame({
			'p1': [rg_file_list[0] for i in xrange(1,len(rg_file_list))],
			'p2': rg_file_list[1:len(rg_file_list)],
			'rg': self.rghat,
			'se': self.rghat_se,
			'z': self.Z,
			'p': self.P
			}).to_string(header=True, index=False)
			
		self.log.log( x )
		self.log.log( '\n' )
		self._print_end_time(args, self.log)
			
	def _print_gencor(self, args, log, rghat, ref_ld_colnames,i, rg_file_list, print_hsq1):
		if print_hsq1:
			self.log.log( '\n' )
			self.log.log( 'Heritability of phenotype 1' )
			self.log.log( '---------------------------' )
			self.log.log(rghat.hsq1.summary(ref_ld_colnames, args.overlap_annot))

		self.log.log( '\n' )
		msg = 'Heritability of phenotype {I}/{N}'.format(I=i+2, N=len(rg_file_list))
		self.log.log(msg)
		self.log.log( ''.join(['-' for i in xrange(len(msg)) ] ))
		self.log.log(rghat.hsq2.summary(ref_ld_colnames, args.overlap_annot))
		self.log.log( '\n' )
		self.log.log( 'Genetic Covariance' )
		self.log.log( '------------------' )
		self.log.log(rghat.gencov.summary(ref_ld_colnames, args.overlap_annot))
		self.log.log( '\n' )
		self.log.log( 'Genetic Correlation' )
		self.log.log( '-------------------' )
		self.log.log(rghat.summary() )
		self.log.log( '\n' )

	def _merge_sumstats_sumstats(self, args, sumstats1, sumstats2, log):
		'''
		Merge two sets of summary statistics and align strand + reference alleles.
		
		This function filters out NA's
				
		'''
		# rename and merge (ideally just concatenate sideways)
		sumstats2.rename(columns={'INC_ALLELE': 'INC_ALLELE2',
			'DEC_ALLELE': 'DEC_ALLELE2', 'N': 'N2',	'CHISQ': 'BETAHAT2'}, inplace=True)
		if 'BETAHAT1' not in sumstats1.columns:
			sumstats1.rename(columns={'CHISQ': 'BETAHAT1', 'N': 'N1'}, inplace=True)
	 		sumstats1['BETAHAT1'] = np.sqrt(sumstats1['BETAHAT1']/sumstats1['N1']) 
			
		x = smart_merge(sumstats1, sumstats2)
		if len(x) == 0:
			raise ValueError('No SNPs remain after merge.')
		# remove NA's
		x['BETAHAT2'] = np.sqrt(x['BETAHAT2']/x['N2'])
		ii = x.BETAHAT1.notnull() & x.BETAHAT2.notnull()
		self.log.log('{N} SNPs with nonmissing values.'.format(N=ii.sum()))
		x = x[ii]
 		if len(x) == 0:
 			raise ValueError('All remaining SNPs have null betahat.')
 	
 		if args.no_check:
			# remove strand ambiguous SNPs
			strand1 = (x.INC_ALLELE+x.DEC_ALLELE).apply(lambda y: STRAND_AMBIGUOUS[y])
			strand2 = (x.INC_ALLELE2+x.DEC_ALLELE2).apply(lambda y: STRAND_AMBIGUOUS[y])
			ii = ~(strand1 | strand2)
			x = x[~(strand1 | strand2)]
			if len(x) == 0:
				raise ValueError('All remaining SNPs are strand ambiguous')
			else:
				msg = 'After removing strand ambiguous SNPs, {N} SNPs remain.'
				log.log(msg.format(N=len(x)))

			# remove SNPs where the alleles do not match
		if not args.no_check_alleles:
			alleles = x.INC_ALLELE+x.DEC_ALLELE+x.INC_ALLELE2+x.DEC_ALLELE2
			match = alleles.apply(lambda y: MATCH_ALLELES[y])
			x = x[match]
			if len(x) == 0:
				raise ValueError('All SNPs have mismatched alleles.')
			else:
				msg = 'After removing SNPs with mismatched alleles, {N} SNPs remain.'
				log.log(msg.format(N=len(x)))
		
		# flip sign of betahat where ref alleles differ
		alleles = x.INC_ALLELE+x.DEC_ALLELE+x.INC_ALLELE2+x.DEC_ALLELE2
		flip = (-1)**alleles.apply(lambda y: FLIP_ALLELES[y])
		x['BETAHAT2'] *= flip
		del x['INC_ALLELE']; del x['DEC_ALLELE']; del x['INC_ALLELE2']; del x['DEC_ALLELE2']
			
		return x
	
	def _rg(self, sumstats_loop, args, log, M_annot, ref_ld_colnames, w_ld_colname, first_hsq=None):
		self.log.log('Estimating genetic correlation.')
		snp_count = len(sumstats_loop); n_annot = len(ref_ld_colnames)
		if snp_count < args.num_blocks:
			num_blocks = snp_count
		else:
			num_blocks = args.num_blocks
			
		self.log.log('Estimating standard errors using a block jackknife with {N} blocks.'.format(N=num_blocks))
		ref_ld = np.matrix(sumstats_loop[ref_ld_colnames]).reshape((snp_count, n_annot))
		w_ld = np.matrix(sumstats_loop[w_ld_colname]).reshape((snp_count, 1))
		M_annot = np.matrix(M_annot).reshape((1, n_annot))
		betahat1 = np.matrix(sumstats_loop.BETAHAT1).reshape((snp_count, 1))
		betahat2 = np.matrix(sumstats_loop.BETAHAT2).reshape((snp_count, 1))
		N1 = np.matrix(sumstats_loop.N1).reshape((snp_count,1))
		N2 = np.matrix(sumstats_loop.N2).reshape((snp_count,1))
		del sumstats_loop
	
		if args.no_intercept:
			args.constrain_intercept = "1,1,0"
	
		if args.constrain_intercept:
			intercepts = args.constrain_intercept.split(',')
			if len(intercepts) != 3:
				msg = 'If using --constrain-intercept with --sumstats_loop-gencor, must specify a ' 
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
		if first_hsq is None:
			rghat = jk.Gencor(betahat1, betahat2, ref_ld, w_ld, N1, N2, M_annot, intercepts,
				args.overlap,	args.rho, num_blocks, return_silly_things=args.return_silly_things,
				slow=args.slow)
		else:
			rghat = jk.Gencor(betahat1, betahat2, ref_ld, w_ld, N1, N2, M_annot, intercepts,
				args.overlap,	args.rho, num_blocks, return_silly_things=args.return_silly_things,
				first_hsq=first_hsq, slow=args.slow)
		
		return rghat
	
	def _filter_chisq(self, args, log, sumstats, N_factor):
		if not args.no_filter_chisq:
			max_N1 = np.max(sumstats['N1'])
			max_N2 = np.max(sumstats['N2'])
			if args.max_chisq is None:
				max_chisq_min = 80
				max_chisq1 = max(N_factor*max_N1, max_chisq_min)
				max_chisq2 = max(N_factor*max_N2, max_chisq_min)
			else:
				max_chisq1, max_chisq2 = args.max_chisq, args.max_chisq
			
			bound1 = np.sqrt(max_chisq1 / max_N1)
			bound2 = np.sqrt(max_chisq2 / max_N2)
			# betahat should be in units of standard deviation
			sumstats = sumstats[(sumstats.BETAHAT1.abs() < bound1) & (sumstats.BETAHAT2.abs() < bound2)]
			if len(sumstats) > 0:
				log_msg = 'After filtering on chi^2 < ({B1},{B2}), {N} SNPs remain.'
				log.log(log_msg.format(N=len(sumstats), B1=round(max_chisq1,2), B2=round(max_chisq2,2)))
			else:
				raise ValueError('After filtering on chi^2, no SNPs remain.')
	
		return sumstats
