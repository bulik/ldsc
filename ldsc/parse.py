from __future__ import division
import numpy as np
import pandas as pd
from scipy.special import chdtri


# input checking functions 

def check_dir(dir):
	c1 = dir != 1
	c2 = dir != -1
	if np.any(np.logical_and(c1, c2)):
		raise ValueError('DIR entry not equal to +/- 1.')


def check_rsid(rsids):
	'''
	Checks that rs numbers are sensible.
	
	'''
	# check for rsid = .
	if np.any(rsids == '.'):
		raise ValueError('Some SNP identifiers are set to . (a dot).')
	
	# check for duplicate rsids
	if np.any(rsids.duplicated('SNP')):
		raise ValueError('Duplicated SNP identifiers.')

	
def check_pvalue(P):
	'''
	Checks that P values are sensible. Nonsense values should have been caught already by 
	coercion to float.
	
	'''
	# check for missing 
	if np.any(np.isnan(P)):
		raise ValueError('Missing P-values')
	
	# check for P outside of the range (0,1]
	if np.max(P) > 1:
		raise ValueError('P-values cannot be > 1.')
	if np.min(P) <= 0:
		raise ValueError('P values cannot be <= 0')


def check_chisq(chisq):
	'''
	Checks that chi-square statistics are sensible. Nonsense values should have been caught
	already by coercion to float.
	
	'''
	if np.any(np.isnan(chisq)):
		raise ValueError('Missing chi-square statistics.')
	
	# check for chisq outside of the range [0,Inf)
	if np.max(chisq) == float('inf'):
		raise ValueError('Infinite chi-square statistics.')
	if np.min(chisq) < 0:
		raise ValueError('Negative chi-square statistics')


def check_maf(maf):
	'''
	Checks that MAFs are sensible. Nonsense values should have been caught already by 
	coercion to float.

	'''
	if np.any(np.isnan(maf)):
		raise ValueError('Missing values in MAF.')
	
	# check for MAF outside of the range (0,1)
	if np.max(maf) >= 1:
		raise ValueError('MAF >= 1.')
	if np.min(maf) <= 0:
		raise ValueError('MAF <= 0.')
	
	
def check_N(N):
	'''
	Checks that sample sizes are sensible. Nonsense values should have been caught already 
	by coercion to int.
	
	'''
	if np.min(N) < 0:
		raise ValueError('Negative N.')


# parsers
def chisq(fh):
	'''
	Parses .chisq files. See docs/file_formats_sumstats.txt
	
	'''
	dtype_dict = {
		'CHR': str,
		'SNP': str,
		'CM': float,
		'BP': int,
		'P': float,
		'CHISQ': float,
		'N': int,
		'MAF': float,
		'INFO': float,
	}
	colnames = open(fh,'rb').readline().split()
	usecols = ['SNP','P','CHISQ','N','MAF','INFO']	
	usecols = [x for x in usecols if x in colnames]
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict)
	except AttributeError as e:
		raise AttributeError('Improperly formatted chisq file: '+ e)

	check_N(x['N'])	
	check_rsid(x['SNP']) 
	
	if 'MAF' in x.columns:
		check_maf(x['MAF'])
		x['MAF'] = np.fmin(x['MAF'], 1-x['MAF'])
	
	if 'P' in x.columns:
		check_pvalue(x['P'])
		x['P'] = chdtri(1, x['P']); 
		x.rename(columns={'P': 'CHISQ'}, inplace=True)
	elif 'CHISQ' in x.columns:
		check_chisq(x['CHISQ'])
	else:
		raise ValueError('.chisq file must have a column labeled either P or CHISQ.')

	return x
	

def betaprod(fh):
	'''
	Parses .betaprod files. See docs/file_formats_sumstats.txt
	
	'''
	dtype_dict = {
		'CHR': str,
		'SNP': str,
		'CM': float,
		'BP': int,
		'P1': float,
		'CHISQ1': float,
		'DIR1': int,
		'N1': int,
		'P2': float,
		'CHISQ2': float,
		'DIR2': int,
		'N2': int,
		'INFO1': float,
		'INFO2': float,
		'MAF1': float,
		'MAF2': float
	}
	colnames = open(fh,'rb').readline().split()
	usecols = [x+str(i) for i in xrange(1,3) for x in ['DIR','P','CHISQ','N','MAF','INFO']]
	usecols.append('SNP')
	usecols = [x for x in usecols if x in colnames]
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict)
	except AttributeError as e:
		raise AttributeError('Improperly formatted betaprod file: '+ e)
		
	check_rsid(x['SNP'])
	
	for i in ['1','2']:
		N='N'+i; P='P'+i; CHISQ='CHISQ'+i; DIR='DIR'+i; MAF='MAF'+i; INFO='INFO'+i
		BETAHAT='BETAHAT'+i
		check_N(x[N])
		check_dir(x[DIR])
		if CHISQ in x.columns:
			check_chisq(x[CHISQ])
			betahat = np.sqrt(x[CHISQ]/x[N]) * x[DIR]
			x[CHISQ] = betahat
			x.rename(columns={CHISQ: BETAHAT}, inplace=True)
		elif P in x.columns:
			check_pvalue(x[P])
			betahat = np.sqrt(chdtri(1, x[P])/x[N])	* x[DIR]
			x[P] = betahat
			x.rename(columns={P: BETAHAT}, inplace=True)
		else:
			raise ValueError('No column named P{i} or CHISQ{i} in betaprod.'.format(i=i))

		del x[DIR]
		if MAF in x.columns:
			check_maf(x[MAF])
			x[MAF]  = np.min(x[MAF], 1-x[MAF])
		
	return x

	
def ldscore(fh):
	'''
	Parses .l2.ldscore files. See docs/file_formats_ld.txt
	
	'''
	fname = fh + '.l2.ldscore'
	x = pd.read_csv(fname, header=0, delim_whitespace=True)
	x = x.drop(['CHR','BP','CM','MAF'],axis=1)
	check_rsid(x['SNP']) 
	print x.columns
	x.ix[:,1:len(x.columns)] = x.ix[:,1:len(x.columns)].astype(float)
	return x


def ldscore22(fh):
	'''
	Parses .l2.ldscore files split across 22 chromosomes (e.g., the output of parallelizing
	ldsc.py --l2 across chromosomes).
	
	'''

	chr_ld = []
	for i in xrange(1,23):
		chr_fh = fh + str(i) + '.l2.ldscore'
		x = pd.read_csv(chr_fh, header=0, delim_whitespace=True)
		x = x.drop(['CHR','BP','CM','MAF'],axis=1)
		chr_ld.append(x)
		
	x = pd.concat(chr_ld)
	x.ix[:,1:len(x.columns)] = x.ix[:,1:len(x.columns)].astype(float)
	check_rsid(x['SNP']) # in case there are duplicated rs#'s on different chromosomes
	return x
	
	
def M(fh):
	'''
	Parses .l2.M files. See docs/file_formats_ld.txt
	
	'''

	fname = fh + '.l2.M'
	x = open(fname, 'r').readline().split()
	x = [float(y) for y in x]
	return x
	
	
def M22(fh):
	'''
	Parses .l2.M files split across 22 chromosomes (e.g., the output of parallelizing
	ldsc.py --l2 across chromosomes).
	
	'''
	chr_M = []
	for i in xrange(1,23):
		chr_fh = fh + str(i)
		chr_M.append(M(chr_fh))
		
	return np.sum(chr_M, axis=0)