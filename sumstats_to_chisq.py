'''
Converts summary stats files to .chisq.gz and alleles files

'''
from __future__ import division
import pandas as pd
import numpy as np
import os
import sys
import gzip
import bz2
import argparse 
from scipy.special import chdtri
from ldscore import sumstats
from ldscore import parse

def get_compression(fh):
	if fh.endswith('gz'):
		compression='gzip'
		openfunc = gzip.open
	elif fh.endswith('bz2'):
		compression = 'bz2'
		openfunc = bz2.BZ2File
	else:
		openfunc = open
		compression=None

	return (openfunc, compression)

# some steps
# capitalize allele names
# remove strand-ambiguous SNPs 	
# capitalize colnames before converting
# figure out how to deal with different Ncas and Ncon
# deal with direction columns

colnames_conversion = {
	# RS NUMBER
	'SNP': 'SNP',
	'MARKERNAME': 'SNP',
	'SNPID': 'SNP',
	
	# P-VALUE
	'P': 'P',
	'PVALUE': 'P',
	'P_VALUE': 	'P',
	'PVAL' : 'P',
	'GC.PVALUE': 'P',

	# ALLELE 1
	'A1': 'A1',
	'ALLELE1': 'A1',
	'EFFECT_ALLELE': 'A1',
	'RISK_ALLELE': 'A1',
	'REFERENCE_ALLELE': 'A1',
	'INC_ALLELE': 'A1',
	

	# ALLELE 2
	'A2' : 'A2',
	'ALLELE2': 'A2',
	'OTHER_ALLELE' : 'A2',
	'NON_EFFECT_ALLELE' : 'A2',
	'DEC_ALLELE': 'A2',

	# N
	'N': 'N',
	'N_CASES': 'N_CAS',
	'N_CONTROLS' : 'N_CON',
	'N_CAS': 'N_CAS',
	'N_CON' : 'N_CON',
	'N_CASE': 'N_CAS',
	'N_CONTROL' : 'N_CON',
	'WEIGHT' : 'N',              # risky
	
	# SIGNED STATISTICS
	'ZSCORE': 'Z',
	'GC.ZSCORE' : 'Z',
	'Z': 'Z',
	'OR': 'OR',
	'BETA': 'BETA',
	'LOG_ODDS': 'LOG_ODDS',
	'EFFECT': 'BETA',
	'EFFECTS': 'BETA',
	
	# INFO
	'INFO': 'INFO',
	
	# MAF
	'FRQ': 'FRQ',
	'MAF': 'FRQ',
	'FRQ_U': 'FRQ',
	'F_U': 'FRQ'
	# don't bother filtering on HM2 CEU MAF
	
}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--sumstats', default=None, type=str,
		help="Input filename.", required=True)
	parser.add_argument('--N', default=None, type=float,
		help="Sample size If this option is not set, will try to infer the sample "
		"size from the input file. If the input file contains a sample size "
		"column, and this flag is set, the argument to this flag has priority.")
	parser.add_argument('--N-cas', default=None, type=float,
		help="Number of cases. If this option is not set, will try to infer the number "
		"of cases from the input file. If the input file contains a number of cases "
		"column, and this flag is set, the argument to this flag has priority.")
	parser.add_argument('--N-con', default=None, type=float,
		help="Number of controls. If this option is not set, will try to infer the number "
		"of controls from the input file. If the input file contains a number of controls "
		"column, and this flag is set, the argument to this flag has priority.")
	parser.add_argument('--out', default=None, type=str,
		help="Output filename prefix.", required=True)
	#parser.add_argument('--gc', default=None, type=float.
	#	help="Second GC correction factor. Will un-do second-level GC correction by "
	#	"multiplying all chi^2 statistics by the argument to --gc.")
	parser.add_argument('--info', default=0.9, type=float,
		help="Minimum INFO score.")
	parser.add_argument('--maf', default=0.01, type=float,
		help="Minimum MAF.")
	parser.add_argument('--daner', default=False, action='store_true',
		help="Use this flag to parse Step	han Ripke's daner* file format.")
	parser.add_argument('--merge', default=None, type=str,
		help="Path to file with a list of SNPs to merge w/ the SNPs in the input file. "
		"Will print the same SNPs in the same order as the --merge file, "
		"with NA's for SNPs in the --merge file and not in the input file." )
	parser.add_argument('--no-alleles', default=False, action="store_true" ,
		help="Don't require alleles. Useful if only unsigned summary statistics are available "
		"and the goal is h2 / partitioned h2 estimation rather than rg estimation.")
	parser.add_argument('--pickle', default=None, action='store_true',
		help="Save .chisq file as python pickle.")
	parser.add_argument('--merge-alleles', default=None, type=str,
		help="Same as --merge, except the file should have three columns: SNP, A1, A2, " 
		"and all alleles will be matched to the --merge-alleles file alleles.")

	args = parser.parse_args()

	if args.merge and args.merge_alleles:
		raise ValueError('--merge and --merge-alleles are not compatible.')

	(openfunc, compression) = get_compression(args.sumstats)
	out_chisq = args.out+'.chisq'
	out_allele = args.out+'.allele'
	colnames = openfunc(args.sumstats).readline().split()

	# also read FRQ_U_* and FRQ_A_* columns from Stephan Ripke's daner* files
	if args.daner:
		frq = filter(lambda x: x.startswith('FRQ_U_'), colnames)[0]
		colnames_conversion[frq] = 'MAF'
	
	usecols = [x for x in colnames if x.upper() in colnames_conversion.keys()]
	dat = pd.read_csv(args.sumstats, delim_whitespace=True, header=0, compression=compression,	
		usecols=usecols)
	print "Read summary statistics for {M} SNPs from {F}.".format(M=len(dat), F=args.sumstats)

	# infer # cases and # controls from daner* column headers
	if args.daner:
		N_con = int(filter(lambda x: x.startswith('FRQ_U_'), colnames)[0].lstrip('FRQ_U_'))
		N_cas = int(filter(lambda x: x.startswith('FRQ_A_'), colnames)[0].lstrip('FRQ_A_'))
		dat['N'] = N_cas + N_con

	# convert colnames
	dat.columns = map(str.upper, dat.columns)
	dat.rename(columns=colnames_conversion, inplace=True)

	# remove NA's
	M = len(dat)
	dat.dropna(axis=0, how="any", inplace=True)
	if len(dat) == 0:
		raise ValueError('All SNPs had at least one missing value.')
	elif len(dat) < M:
		print "Removed {M} SNPs with missing values.".format(M=(M-len(dat)))

	# filter p-vals
	dat = dat[~np.isnan(dat.P)]
	dat = dat[dat.P>0]

	# N
	if args.N:
		dat['N'] = args.N
	elif args.N_cas and args.N_con:
		dat['N'] = args.N_cas + args.N_con
	elif 'N_CAS' in dat.columns and 'N_CON' in dat.columns:
		N = dat.N_CAS + dat.N_CON
		P = dat.N_CAS / N
		ii = N == N.max()
		P_max = P[ii].mean()
		print "Using P_max = {P}.".format(P=P_max)
		dat['N'] = N * P /	 P_max
		del dat['N_CAS']
		del dat['N_CON']
	elif 'N' in dat.columns:
		pass
	else:
		raise ValueError('No N specified.')

	# filter on INFO
	if 'INFO' in dat.columns:
		dat = dat[dat.INFO > args.info]
		print 'After filtering on INFO > {C}, {M} SNPs remain.'.format(C=args.info,	M=len(dat))
	
	# convert FRQ to MAF and filter on MAF
	if 'FRQ' in dat.columns:
		dat.FRQ = np.minimum(dat.FRQ, 1-dat.FRQ)
		dat.rename(columns={'FRQ': 'MAF'}, inplace=True)
		dat = dat[dat.MAF > args.maf]
		print 'After filtering on MAF > {C}, {M} SNPs remain.'.format(C=args.maf,	M=len(dat))

	# check uniqueness of rs numbers
	# if not unique, uniquify by taking the first row for each rs number
	old = len(dat)
	dat.drop_duplicates('SNP', inplace=True)
	new = len(dat)
	if old != new:
		print "Removed {N} SNPs with duplicated rs numbers.".format(N=old-new)

	# convert p-values to chi^2
	chisq = chdtri(1, dat['P'])
	dat.P = chisq
	dat.rename(columns={'P': 'CHISQ'}, inplace=True)

	# everything with alleles here
	if 'A1' in dat.columns and 'A2' in dat.columns and not args.no_alleles:

		# capitalize alleles
		dat.A1 = dat.A1.apply(lambda y: y.upper())
		dat.A2 = dat.A2.apply(lambda y: y.upper())

		# filter out indels
		ii = (dat.A1 == 'A') | (dat.A1 == 'T') | (dat.A1 == 'C') | (dat.A1 == 'G')
		ii = ii & (dat.A2 == 'A') | (dat.A2 == 'T') | (dat.A2 == 'C') | (dat.A2 == 'G')
		if ii.sum() < len(dat):
			print "Removed {M} variants not coded A/C/T/G.".format(M=(len(dat)-ii.sum()))
		
		dat = dat[ii]
	
		# remove strand ambiguous SNPs
		strand = (dat.A1 + dat.A2).apply(lambda y: sumstats.STRAND_AMBIGUOUS[y])
		dat = dat[~strand]
		if len(dat) == 0:
			raise ValueError('All remaining SNPs are strand ambiguous')
		else:
			msg = 'After removing strand ambiguous SNPs, {N} SNPs remain.'
			print msg.format(N=len(dat))
		
		# signed summary stat and alleles
		if 'OR' in dat.columns:
			dat.OR = dat.OR.convert_objects(convert_numeric=True)
			dat = dat[~np.isnan(dat.OR)]
			flip = dat.OR < 1
		elif 'Z' in dat.columns:
			dat.Z = dat.Z.convert_objects(convert_numeric=True)
			dat = dat[~np.isnan(dat.Z)]
			flip = dat.Z < 0
		elif 'BETA' in dat.columns:
			dat.BETA = dat.BETA.convert_objects(convert_numeric=True)
			dat = dat[~np.isnan(dat.BETA)]
			flip = dat.BETA < 0
		elif 'LOG_ODDS' in dat.columns:
			dat.LOG_ODDS = dat.LOG_ODDS.convert_objects(convert_numeric=True)
			dat = dat[~np.isnan(dat.LOG_ODDS)]
			flip = dat.LOG_ODDS < 0
		else: # assume A1 is trait increasing allele and print a warning
			print 'Warning: no signed summary stat found. Assuming A1 is risk/increasing allele.'
			flip = pd.Series(False)
	
		# convert A1 and A2 to INC_ALLELE and DEC_ALLELE
		INC_ALLELE = dat.A1
		DEC_ALLELE = dat.A2

		if flip.any():
			x = dat.A1[flip]
			INC_ALLELE[flip] = dat.A2[flip]
			DEC_ALLELE[flip] = x
				
		dat['INC_ALLELE'] = INC_ALLELE
		dat['DEC_ALLELE'] = DEC_ALLELE
		del dat['A1']; del dat['A2']
	
		# merge with --merge-alleles
		if args.merge_alleles:
			(openfunc, compression) = get_compression(args.merge_alleles)
			merge_alleles = pd.read_csv(args.merge_alleles, compression=compression, header=0, 
				delim_whitespace=True)
			print merge_alleles.columns
			if len(merge_alleles.columns) == 1 | np.all(merge_alleles.columns != ["SNP","A1","A2"]):
				raise ValueError('--merge-alleles must have columns SNP, A1, A2.')
		
			merge_alleles.A1 = merge_alleles.A1.apply(lambda y: y.upper())
			merge_alleles.A2 = merge_alleles.A2.apply(lambda y: y.upper())
# 			print merge_alleles.head()
# 			print merge_alleles.tail()
# 			print merge_alleles.dtypes
# 			print len(merge_alleles)
# 			# WARNING: dat now contains many NaN values
# 			#dat = dat[dat.SNP.isin(merge_alleles.SNP)]
# 			dat1=pd.merge(dat, merge_alleles, how="inner", on="SNP", sort=False).reset_index(drop=True)
# 			dat=pd.merge(merge_alleles, dat, how="inner", on="SNP", sort=False).reset_index(drop=True)

			dat = pd.merge(merge_alleles, dat, how='inner', on='SNP', sort=False).reset_index(drop=True)
 			ii = dat.N.notnull()
			print 'After LOJ on --merge-alleles, we have {N} SNPs of which {M} have nonmissing data'.format(N=len(dat), M=ii.sum())
 			alleles = dat.INC_ALLELE[ii] + dat.DEC_ALLELE[ii] + dat.A1[ii] + dat.A2[ii]
 			try:
 				match = alleles.apply(lambda y: sumstats.MATCH_ALLELES[y])
 			except KeyError as e:
 				msg = "Does your --merge-alleles file contain indels or strand ambiguous SNPs?"
 				print msg
 				raise 
 				
 			x = dat[ii]
 			jj = pd.Series([False	 for j in xrange(len(dat))])
			jj[ii] = match
 			dat.N[~jj] = 1
 			dat.CHISQ[~jj] = 1
 
 			if len(dat) == 0:
 				raise ValueError('All SNPs have mismatched alleles.')
 			else:
 				msg = 'After removing SNPs with mismatched alleles, {N} SNPs remain. '
 				msg += 'of which {M} have non-missing data.'
 				print msg.format(N=len(dat), M=dat.CHISQ.notnull().sum())
		
			dat = dat.drop(['A1','A2'], axis=1)

	elif not args.no_alleles:
		raise ValueError('Could not find A1 and A2 columns in --sumstats.')

	# merge with --merge file 
	# keep the same SNPs in the same order (w/ NA's) 
	# so that ldsc can use concat instead of merge
	if args.merge:
		(openfunc, compression) = get_compression(args.merge)
		merge_snplist = pd.read_csv(args.merge, compression=compression, header=None)
		merge_snplist.columns=['SNP']

		print 'Read list of {M} SNPs to retain from {F}.'.format(M=len(merge_snplist),
			F=args.merge)
	
		dat = pd.merge(merge_snplist, dat, on='SNP', how='left', sort=False).reset_index(drop=True)
		remain = dat.CHISQ.notnull().sum()
		print 'After merging with --merge SNPs, {M} SNPs remain.'.format(M=remain)

	# write chisq file
	chisq_colnames = [c for c in ['SNP','INFO','N','CHISQ','MAF','INC_ALLELE','DEC_ALLELE'] 
		if c in dat.columns]
	if not args.pickle:
		print 'Writing chi^2 statistics for {M} SNPs to {F}.'.format(M=len(dat), F=out_chisq+'.gz')
		dat.ix[:,chisq_colnames].to_csv(out_chisq, sep="\t", index=False)
		os.system('gzip -f {F}'.format(F=out_chisq))
	else:
		print 'Writing chi^2-statistics for {M} SNPs to {F}.'.format(M=len(dat), F=out_chisq+'.pickle')
		out_chisq += '.pickle'
		dat.ix[:,chisq_colnames].reset_index(drop=True).to_pickle(out_chisq)

	# write metadata
	np.set_printoptions(precision=4)
	pd.set_option('precision', 4)
	pd.set_option('display.max_rows', 100000)

	metadat_fh = args.out+'.chisq.metadata'
	mfh = open(metadat_fh, 'w')
	print 'Writing metadata to {F}.'.format(M=len(dat), F=metadat_fh)
	# mean chi^2 
	mean_chisq = np.mean(chisq)
	print >>mfh, 'Mean Chi-Square = ' + str(round(mean_chisq,3))
	if mean_chisq < 1.02:
		print "WARNING: mean chi^2 may be too small."

	# lambda GC 
	lambda_gc = np.median(chisq) / 0.4549
	print >>mfh, 'Lambda GC = ' + str(round(lambda_gc,3))

	# min p-value
	print >>mfh, 	'Max chi^2 = ' + str(np.matrix(np.max(dat.CHISQ))).replace('[[','').replace(']]','').replace('  ',' ')

	# most significant SNPs
	print >>mfh, "Genome-wide significant SNPs:"
	ii = dat.CHISQ > 29
	print >>mfh, dat[ii]
	mfh.close()