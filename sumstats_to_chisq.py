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
from scipy.stats import chi2
from ldscore import sumstats
from ldscore import parse
from ldsc import MASTHEAD
import time


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

colnames_conversion = {
	# RS NUMBER
	'SNP': 'SNP',
	'MARKERNAME': 'SNP',
	'SNPID': 'SNP',
	'RS' : 'SNP',
	'RSID': 'SNP',
	'RS_NUMBER': 'SNP',
	'RS-NUMBER': 'SNP',
	'RS_NUMBERS': 'SNP',
	'RS-NUMBERS': 'SNP',
	
	# P-VALUE
	'P': 'P',
	'PVALUE': 'P',
	'P_VALUE': 	'P',
	'PVAL' : 'P',
	'P-VAL' : 'P',
	'P-VALUE':	'P',
	'GC.PVALUE': 'P',

	# ALLELE 1
	'A1': 'A1',
	'ALLELE1': 'A1',
	'ALLELE_1': 'A1',
	'ALLELE-1': 'A1',
	'EFFECT_ALLELE': 'A1',
	'RISK_ALLELE': 'A1',
	'REFERENCE_ALLELE': 'A1',
	'INC_ALLELE': 'A1',
	'EA': 'A1',

	# ALLELE 2
	'A2' : 'A2',
	'ALLELE2': 'A2',
	'ALLELE_2': 'A2',
	'ALLELE-2': 'A2',
	'OTHER_ALLELE' : 'A2',
	'NON_EFFECT_ALLELE' : 'A2',
	'DEC_ALLELE': 'A2',
	'NEA': 'A2',
#	'REF_ALLELE': 'A2',
#	'REFERENCE_ALLELE': 'A2',

	# N
	'N': 'N',
	'NCASE': 'N_CAS',
	'N_CASES': 'N_CAS',
	'N_CONTROLS' : 'N_CON',
	'N_CAS': 'N_CAS',
	'N_CON' : 'N_CON',
	'N_CASE': 'N_CAS',
	'NCONTROL': 'N_CON',
	'N_CONTROL': 'N_CON',
	'WEIGHT' : 'N',              # metal does this. possibly risky.
	
	# SIGNED STATISTICS
	'ZSCORE': 'Z',
	'GC.ZSCORE' : 'Z',
	'Z': 'Z',
	'OR': 'OR',
	'BETA': 'BETA',
	'LOG-ODDS': 'LOG_ODDS',
	'LOG_ODDS': 'LOG_ODDS',
	'EFFECT': 'BETA',
	'EFFECTS': 'BETA',
	
	# INFO
	'INFO': 'INFO',
	
	# MAF
	'FRQ': 'FRQ',
	'MAF': 'FRQ',
	'FRQ_U': 'FRQ',
	'F_U': 'FRQ',
	'CEUAF': 'FRQ',
	'CEU_AF': 'FRQ'
}
#@profile
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--sumstats', default=None, type=str,
		help="Input filename.")
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
		help="Output filename prefix.")
	parser.add_argument('--info', default=0.9, type=float,
		help="Minimum INFO score.")
	parser.add_argument('--maf', default=0.01, type=float,
		help="Minimum MAF.")
	parser.add_argument('--daner', default=False, action='store_true',
		help="Use this flag to parse Step	han Ripke's daner* file format.")
	parser.add_argument('--merge', default=None, type=str,
		help="Path to file with a list of SNPs to merge w/ the SNPs in the input file. "
		"Will log.log( the same SNPs in the same order as the --merge file, "
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
	
	log = sumstats.Logger(args.out + '.log')
	
	if not (args.sumstats and args.out):
		raise ValueError('--sumstats and --out are required.')
	
	defaults = vars(parser.parse_args(''))
	opts = vars(args)
	non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
	
	header = MASTHEAD
	header += "\nOptions: \n"
	options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
	header += '\n'.join(options).replace('True','').replace('False','')
	header += '\n'
	log.log( header )

	log.log('Beginning conversion at {T}'.format(T=time.ctime()))
	start_time = time.time()

	print 'Writing log to {F}'.format(F=log.log_name)
	if args.merge and args.merge_alleles:
		raise ValueError('--merge and --merge-alleles are not compatible.')

	(openfunc, compression) = get_compression(args.sumstats)
	out_chisq = args.out+'.chisq'
	colnames = openfunc(args.sumstats).readline().split()

	# also read FRQ_U_* and FRQ_A_* columns from Stephan Ripke's daner* files
	if args.daner:
		frq = filter(lambda x: x.startswith('FRQ_U_'), colnames)[0]
		colnames_conversion[frq] = 'MAF'
	
	usecols = [x for x in colnames if x.upper() in colnames_conversion.keys()]
	dat = pd.read_csv(args.sumstats, delim_whitespace=True, header=0, compression=compression,	
		usecols=usecols, na_values='.')
	log.log( "Read summary statistics for {M} SNPs from {F}.".format(M=len(dat), F=args.sumstats))

	# infer # cases and # controls from daner* column headers
	if args.daner:
		N_con = int(filter(lambda x: x.startswith('FRQ_U_'), colnames)[0].lstrip('FRQ_U_'))
		N_cas = int(filter(lambda x: x.startswith('FRQ_A_'), colnames)[0].lstrip('FRQ_A_'))
		dat['N'] = N_cas + N_con
		log.log( 'Inferred that N_cas = {N} from the FRQ_A column.'.format(N=N_cas))
		log.log( 'Inferred that N_con = {N} from the FRQ_U column.'.format(N=N_con))
		
	# convert colnames
	dat.columns = map(str.upper, dat.columns)
	dat.rename(columns=colnames_conversion, inplace=True)
	
	# check uniqueness of rs numbers & remove SNPs w/ rs == '.'
	old_len = len(dat)
	dat = dat.drop_duplicates('SNP')
	new_len = len(dat)
	msg = 'Removed {M} SNPs with duplicated rs numbers ({N} SNPs remain).'
	log.log(msg.format(M=old_len-new_len, N=new_len))
	if new_len == 0:
		raise ValueError('No SNPs remain.')
		
	# remove NA's
	old_len = len(dat)
	dat.dropna(axis=0, how="any", inplace=True)
	new_len = len(dat)
	msg = 'Removed {M} SNPs with missing values ({N} SNPs remain).'
	log.log(msg.format(M=old_len-new_len, N=new_len))
	if new_len == 0:
		raise ValueError('No SNPs remain.')

	
	if dat.P.dtype != 'float':
		log.log( 'Coercing P-value column to float.')
		dat.P = dat.P.astype('float')		

 	if args.merge:
		pass
# 		(openfunc, compression) = get_compression(args.merge)
# 		merge_snplist = pd.read_csv(args.merge, compression=compression, header=None)
# 		merge_snplist.columns=['SNP']
# 
# 		log.log( 'Read list of {M} SNPs to retain from {F}.'.format(M=len(merge_snplist),
# 			F=args.merge)
# 			
# 		dat = dat[dat.SNP.isin(merge_alleles.SNP)].reset_index(drop=True)
# 		dat = pd.merge(merge_snplist, dat, on='SNP', how='left', sort=False).reset_index(drop=True)
# 		remain = dat.CHISQ.notnull().sum()
# 		log.log( 'After merging with --merge SNPs, {M} SNPs remain.'.format(M=remain)


	elif args.merge_alleles:
		log.log('Reading list of SNPs for allele merge from {F}'.format(F=args.merge_alleles))
		(openfunc, compression) = get_compression(args.merge_alleles)
		merge_alleles = pd.read_csv(args.merge_alleles, compression=compression, header=0, 
			delim_whitespace=True)
		if len(merge_alleles.columns) == 1 | np.all(merge_alleles.columns != ["SNP","A1","A2"]):
			raise ValueError('--merge-alleles must have columns SNP, A1, A2.')
		
		log.log('Read {N} SNPs for allele merge.'.format(N=len(merge_alleles)))
		merge_alleles.A1 = merge_alleles.A1.apply(lambda y: y.upper())
		merge_alleles.A2 = merge_alleles.A2.apply(lambda y: y.upper())
		old_len = len(dat)
		dat = dat[dat.SNP.isin(merge_alleles.SNP)].reset_index(drop=True)
		new_len = len(dat)
		msg = 'Removed {M} SNPs in --sumstats not in --merge-alleles ({N} SNPs remain).'
		log.log(msg.format(M=old_len-new_len, N=new_len))
		if new_len == 0:
			raise ValueError('No SNPs remain.')
			
	# N
	if args.N:
		dat['N'] = args.N
		log.log( 'Using N = {N}'.format(N=args.N))

	elif args.N_cas and args.N_con:
		dat['N'] = args.N_cas + args.N_con
		msg = 'Using N_cas = {N1}; N_con = {N2}' 
		log.log( msg.format(N1=args.N_cas, N2=args.N_con))
	
	elif 'N_CAS' in dat.columns and 'N_CON' in dat.columns:
		log.log( 'Reading sample size from the N_cas and N_con columns.')
		msg = 'Median N_cas = {N1}; Median N_con = {N2}'
		log.log(msg.format(N1=round(np.median(dat.N_cas), 0), N2=round(np.median(dat.N_con),0)))
		N = dat.N_CAS + dat.N_CON
		P = dat.N_CAS / N
		ii = N == N.max()
		P_max = P[ii].mean()
		log.log( "Using max sample prevalence = {P}.".format(P=round(P_max,2)))
		dat['N'] = N * P /	 P_max
		dat.drop(['N_cas', 'N_con'], inplace=True, axis=1)
	
	elif 'N' in dat.columns:
		log.log( 'Reading sample size from the N column. Median N = {N}'.format(N=round(np.median(dat.N), 0)))

	else:
		raise ValueError('No N specified.')
		
	# filter p-vals
	old_len = len(dat)
	ii = (dat.P > 0) & (dat.P <= 1)
	new_len = ii.sum()
	msg = 'Removed {M} SNPs with p-values outside of (0,1] ({N} SNPs remain).'
	log.log(msg.format(M=old_len-new_len, N=new_len))
	if new_len == 0:
		raise ValueError('No SNPs remain.')

	# filter on INFO
	if 'INFO' in dat.columns:
		ii = ii & (dat.INFO > args.info)
		old_len = new_len
		new_len = ii.sum()
		msg = 'Removed {M} SNPs with INFO <= {C} ({N} SNPs remain).'
		log.log( msg.format(C=args.info,	N=new_len, M=old_len-new_len))
		if new_len == 0:
			raise ValueError('No SNPs remain.')

		dat.drop('INFO', inplace=True, axis=1)
	
	# convert FRQ to MAF and filter on MAF
	if 'FRQ' in dat.columns:
		dat.FRQ = np.minimum(dat.FRQ, 1-dat.FRQ)
		ii = ii & (dat.FRQ > args.maf)
		old_len = new_len
		new_len = ii.sum()		
		msg = 'Removed {M} SNPs with MAF <= {C} ({N} SNPs remain).'
		log.log( msg.format(C=args.maf,	N=new_len, M=old_len-new_len))
		if new_len == 0:
			raise ValueError('No SNPs remain.')

		dat.drop('FRQ', inplace=True, axis=1)
		
	dat = dat[ii]
	# convert p-values to chi^2
	dat.P = chi2.isf(dat.P, 1)
	dat.rename(columns={'P': 'CHISQ'}, inplace=True)

	# everything with alleles here
	if 'A1' in dat.columns and 'A2' in dat.columns and not args.no_alleles:

		# capitalize alleles
		dat.A1 = dat.A1.apply(lambda y: y.upper())
		dat.A2 = dat.A2.apply(lambda y: y.upper())

		# filter out indels
		ii = dat.A1.isin(['A','T','C','G'])
		ii = ii & dat.A2.isin(['A','T','C','G'])
		old_len = len(dat)
		new_len = ii.sum()
		if new_len == 0:
			raise ValueError('Every SNP was not coded A/C/T/G. Something is wrong.')
		else:
			msg = 'Removed {M} variants not coded A/C/T/G ({N} SNPs remain)'
			log.log( msg.format(C=args.info,	N=new_len, M=old_len-new_len))

		dat = dat[ii]
	
		# remove strand ambiguous SNPs
		strand = (dat.A1 + dat.A2).apply(lambda y: sumstats.STRAND_AMBIGUOUS[y])
		dat = dat[~strand]
		old_len = new_len
		new_len = len(dat)
		if new_len == 0:
			raise ValueError('All remaining SNPs are strand ambiguous')
		else:
			msg = 'Removed {M} strand ambiguous SNP ({N} SNPs remain)'
			log.log( msg.format(C=args.info,	N=new_len, M=old_len-new_len))
		
		# signed summary stat and alleles
		if 'OR' in dat.columns:
			log.log('Using OR (odds ratio) as the directional summary statistic.')
			check = np.mean(dat.OR)
			if np.abs(check-1) > 0.1:
				msg = 'WARNING: mean OR is {M} (should be close to 1). This column may be mislabeled.'
				log.log( msg.format(M=round(check,2)))

			dat.OR = dat.OR.convert_objects(convert_numeric=True)
			flip = dat.OR < 1
			dat.drop('OR', inplace=True, axis=1)

		elif 'Z' in dat.columns:
			log.log('Using Z (Z-score) as the directional summary statistic.')
			check = np.mean(dat.Z)
			if np.abs(check) > 0.1:
				msg = 'WARNING: mean Z is {M} (should be close to 0). This column may be mislabeled.'
				log.log( msg.format(M=round(check,2)))

			dat.Z = dat.Z.convert_objects(convert_numeric=True)
			flip = dat.Z < 0
			dat.drop('Z', inplace=True, axis=1)
			
		elif 'BETA' in dat.columns:			
			log.log('Using BETA as the directional summary statistic.')
			check = np.mean(dat.BETA)
			if np.abs(check) > 0.1:
				msg = 'WARNING: mean BETA is {M} (should be close to 0). This column may be mislabeled.'
				log.log( msg.format(M=round(check,2)))

			dat.BETA = dat.BETA.convert_objects(convert_numeric=True)
			flip = dat.BETA < 0
			dat.drop('BETA', inplace=True, axis=1)
			
		elif 'LOG_ODDS' in dat.columns:
			log.log('Using Log odds  as the directional summary statistic.')
			check = np.mean(dat.LOG_ODDS)
			if np.abs(check) > 0.1:
				msg = 'WARNING: mean Log odds is {M} (should be close to 0). This column may be mislabeled.'
				log.log( msg.format(M=round(check,2)))

			dat.LOG_ODDS = dat.LOG_ODDS.convert_objects(convert_numeric=True)
			flip = dat.LOG_ODDS < 0
			dat.drop('LOG_ODDS', inplace=True, axis=1)
			
		else: # assume A1 is trait increasing allele and print a warning
			log.log( 'Warning: no signed summary stat found. Assuming A1 is risk/increasing allele.')
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
		dat.drop(['A1','A2'], inplace=True, axis=1)
	
		# merge with --merge-alleles
		if args.merge_alleles:
			
			'''
			WARNING: dat now contains a bunch of NA's~
			Note: dat now has the same SNPs in the same order as --merge alleles.
			'''
			
			dat = pd.merge(merge_alleles, dat, how='left', on='SNP', sort=False).reset_index(drop=True)
 			ii = dat.N.notnull()
 			alleles = dat.INC_ALLELE[ii] + dat.DEC_ALLELE[ii] + dat.A1[ii] + dat.A2[ii]
 			try:
 				# true iff the alleles match; false --> throw out
 				match = alleles.apply(lambda y: sumstats.MATCH_ALLELES[y]) 
 			except KeyError as e:
 				msg = 'Could not match alleles between --sumstats and --merge-alleles.\n'
 				msg += 'Does your --merge-alleles file contain indels or strand ambiguous SNPs?'
 				log.log( msg )
 				raise ValueError(msg)
 				
 			x = dat[ii]
 			jj = pd.Series(np.zeros(len(dat),dtype=bool))
			jj[ii] = match
			# set all SNPs that were already NA or had funky alleles to NA
 			dat.N[~jj] = float('nan')
 			dat.CHISQ[~jj] = float('nan')
			dat.INC_ALLELE[~jj] = float('nan')
			dat.DEC_ALLELE[~jj] = float('nan')
			old_len = new_len
			new_len = jj.sum()
 			if len(dat) == 0:
 				raise ValueError('All SNPs have alleles that are discordant between --sumstats and --merge-alleles.')
 			else:
 				msg = 'Removed {M} SNPs whose alleles did not match --merge-alleles ({N} SNPs remain).'
				log.log( msg.format(N=new_len, M=old_len-new_len))
		
			dat = dat.drop(['A1','A2'], axis=1)

	elif not args.no_alleles:
		raise ValueError('Could not find A1 and A2 columns in --sumstats.')
		
	# write chisq file
	chisq_colnames = [c for c in ['SNP','INFO','N','CHISQ','MAF','INC_ALLELE','DEC_ALLELE'] 
		if c in dat.columns]
	if not args.pickle:
		msg = 'Writing chi^2 statistics for {M} SNPs ({N} of which have nonmissing chi^2) to {F}.'
		log.log( msg.format(M=len(dat), F=out_chisq+'.gz', N=dat.N.notnull().sum()))
		dat.ix[:,chisq_colnames].to_csv(out_chisq, sep="\t", index=False)
		os.system('gzip -f {F}'.format(F=out_chisq))
	else:
		msg = 'Writing chi^2 statistics for {M} SNPs ({N} of which have nonmissing chi^2) to {F}.'
		log.log( msg.format(M=len(dat), F=out_chisq+'.pickle', N=dat.N.notnull().sum()))
		out_chisq += '.pickle'
		dat.ix[:,chisq_colnames].reset_index(drop=True).to_pickle(out_chisq)

	log.log( '\n' )
	log.log('Conversion finished at {T}'.format(T=time.ctime()) )
	time_elapsed = round(time.time()-start_time,2)
	log.log('Total time elapsed: {T}'.format(T=sumstats.sec_to_str(time_elapsed)))
	log.log( '\n' )
	log.log('Printing metadata')
	# write metadata
	np.set_printoptions(precision=4)
	pd.set_option('precision', 4)
	pd.set_option('display.max_rows', 100000)

	# mean chi^2 
	mean_chisq = dat.CHISQ.mean()
	log.log( 'Mean chi^2 = ' + str(round(mean_chisq,3)))
	if mean_chisq < 1.02:
		log.log( "WARNING: mean chi^2 may be too small.")

	# lambda GC 
	lambda_gc = np.median(dat.CHISQ) / 0.4549
	log.log( 'Lambda GC = ' + str(round(lambda_gc,3)))

	# min p-value
	log.log('Max chi^2 = ' + str(np.matrix(np.max(dat.CHISQ))).replace('[[','').replace(']]','').replace('  ',' '))

	# most significant SNPs
	ii = dat.CHISQ > 29
 	if ii.sum() > 0:
		log.log( "Genome-wide significant SNPs:\n")
		log.log( dat[ii])
	else:
		log.log('No genome-wide significant SNPs')
		log.log('NB some gwsig SNPs may have been removed after various filtering steps.')