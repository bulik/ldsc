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

if sys.version_info < (2,7):
  raise ValueError('You forgot to use Python-2.7.')

def get_compression(fh):
	if fh.endswith('gz'):
		compression='gzip'
		openfunc = gzip.open
	elif fh.endswith('bz2'):
		compression = 'bz2'
		openfunc =  bz2.BZ2File
	else:
		openfunc = open
		compression=None
		print "Compress your summary stats, bro! Terabytes = $$$"

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
	'PVAL' : 'P',
	'GC.PVALUE': 'P',

	# ALLELE 1
	'A1': 'A1',
	'ALLELE1': 'A1',
	'EFFECT_ALLELE': 'A1',
	'RISK_ALLELE': 'A1',

	# ALLELE 2
	'A2' : 'A2',
	'ALLELE2': 'A2',
	'OTHER_ALLELE' : 'A2',
	'NON_EFFECT_ALLELE' : 'A2',

	# N
	'N': 'N',
	'N_CASES': 'N_CAS',
	'N_CONTROLS' : 'N_CON',
	'N_CAS': 'N_CAS',
	'N_CON' : 'N_CON',
	'WEIGHT' : 'N',              # risky
	
	# SIGNED STATISTICS
	'ZSCORE': 'Z',
	'GC.ZSCORE' : 'Z',
	'Z': 'Z',
	'OR': 'OR',
	'BETA': 'BETA',
	
	# INFO
	'INFO': 'INFO',
	
	# MAF
	'FRQ': 'FRQ',
	'MAF': 'FRQ',
	# don't bother filtering on HM2 CEU MAF
	
}


parser = argparse.ArgumentParser()
parser.add_argument('--sumstats', default=None, type=str)
parser.add_argument('--N', default=None, type=float)
parser.add_argument('--N-cas', default=None, type=float)
parser.add_argument('--N-con', default=None, type=float)
parser.add_argument('--out', default=None, type=str)
parser.add_argument('--gc', default=None, type=float)
parser.add_argument('--info', default=0.9, type=float)
parser.add_argument('--maf', default=0.01, type=float)
parser.add_argument('--daner', default=False, action='store_true')
parser.add_argument('--merge', default='/humgen/atgu1/fs03/data/hm3.snplist.gz',
	type=str)
args = parser.parse_args()


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

# capitalize alleles
dat.A1 = dat.A1.apply(lambda y: y.upper())
dat.A2 = dat.A2.apply(lambda y: y.upper())

# filter p-vals
dat = dat[~np.isnan(dat.P)]
dat = dat[dat.P>0]

# merge with --merge file 
if args.merge:
	(openfunc, compression) = get_compression(args.merge)
	merge_snplist = pd.read_csv(args.merge, compression=compression, header=None)
	merge_snplist.columns=['SNP']
	print 'Read list of {M} SNPs to retain from {F}.'.format(M=len(merge_snplist),
		F=args.merge)
	dat = dat[dat.SNP.isin(merge_snplist.SNP)]
	print 'After merging with --merge SNPs, {M} SNPs remain.'.format(M=len(dat))

# N
if args.N:
	dat['N'] = args.N
elif args.N_cas and args.N_con:
	dat['N'] = args.N_cas + args.N_con
elif 'N_CAS' in dat.columns:
	dat['N'] = dat.N_CAS + dat.N_CON
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
else: # assume A1 is trait increasing allele and print a warning
	print 'Warning: no signed summary stat found. Assuming A1 is risk/increasing allele.'
	flip = pd.Series(False)
	
INC_ALLELE = dat.A1
if flip.any():
	INC_ALLELE[flip] = dat.A2[flip]
dat['INC_ALLELE'] = INC_ALLELE

# chisq.gz
print 'Writing P-values for {M} SNPs to {F}.'.format(M=len(dat), F=out_chisq+'.gz')
chisq_colnames = [c for c in ['SNP','INFO','N','P','MAF'] if c in dat.columns]
dat.ix[:,chisq_colnames].to_csv(out_chisq, sep="\t", index=False)
os.system('gzip -f {F}'.format(F=out_chisq))

# allele.gz
print 'Writing alleles for {M} SNPs to {F}.'.format(M=len(dat), F=out_allele+'.gz')
dat.ix[:,['SNP','INC_ALLELE']].to_csv(out_allele,sep='\t',index=False)
os.system('gzip -f {F}'.format(F=out_allele))