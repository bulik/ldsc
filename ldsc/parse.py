from __future__ import division
import numpy as np
import pandas as pd
from scipy.special import chdtri


# input checking functions 

def check_dir(dir):
	c1 = dir != 1
	c2 = dir != -1
	if np.any(np.logical_and(c1, c2)):
		raise ValueError('DIR entry not equal to +/- 1.)


def check_rsid(rsids):

	# check for nan 
	if np.any(np.isnan(rsids)):
		raise ValueError('Some SNP identifiers are nan.')
	
	# check for rsid = .
	if np.any(rsids == '.'):
		raise ValueError('Some SNP identifiers are set to . (a dot).'
	
	# check for duplicate rsids
	if np.any(rsids.duplicated('SNP'):
		raise ValueError('Duplicated SNP identifiers.')

	
def check_pvalue(P):

	# nonsense values should have been caught already by coercion to float
	
	# check for missing 
	if np.any(np.isnan(P)):
		raise ValueError('Missing P-values')
	
	# check for P outside of the range (0,1]
	if np.max(P) > 1:
		raise ValueError('P-values cannot be > 1.')
	if np.min(P) <= 0:
		raise ValueEror('P values cannot be <= 0')


def check_chisq(chisq):

	# nonsense values should have been caught already by coercion to float
	if np.any(np.isnan(chisq)):
		raise ValueError('Missing chi-square statistics.')
	
	# check for chisq outside of the range [0,Inf)
	if np.max(chisq) == float('inf'):
		raise ValueError('Infinite chi-square statistics.')
	if np.min(chisq) < 0:
		raise ValueError('Negative chi-square statistics')


def check_maf(maf):
	
	# nonsense values should have been caught already by coercion to float
	if np.any(np.isnan(maf)):
		raise ValueError('Missing values in MAF.')
	
	# check for MAF outside of the range (0,1)
	if np.max(chisq) >= 1:
		raise ValueError('MAF > 1.')
	if np.min(chisq) <= 0:
		raise ValueError('MAF < 0.)
	
	
def check_N(N):
	# coercion to int should have already taken care of nonsense
	
	if np.min(N) < 0:
		raise ValueError('Negative N.')


# parsers

def chisq(fh):
	dtype_dict = {
#		'CHR': str,
		'SNP': str,
#		'CM': float,
#		'BP', int,
		'P', float,
		'CHISQ', float,
		'N', int,
		'MAF', float,
		'INFO', float,
	}
	usecols = dtype_dict.keys()
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict)
	except AttributeError as e:
		raise AttributeError('Improperly formatted chisq file: '+ e)
	
	msg = 'Expected column {I} of {F} to be {C}, got {W}' 
	#for i, c in ['CHR','SNP','CM','BP','N']:
	for i, c in ['SNP','N']
		if x.columns[i] != c:
			raise	ValueError(msg.format(i=i,F=fh,C=c,W=x.columns[i])

	check_N(x['N'])
	check_rsid(x['SNP']) 
	if 'MAF' in x.colnames:
		check_maf(x['MAF'])
		x['MAF'] = np.min(x['MAF'], 1-x['MAF'])
	
	if x.columns[4] == 'P':
		check_pvalue(x['P'])
		x['P'] = chdtri(1, x['P']); 
		x.columns[4] = 'CHISQ'
	elif x.columns[4] == 'CHISQ':
		check_chisq(x['CHISQ'])
	else:
		msg = 'Expected column 5 of {F} to be P or CHISQ, got {W}' 
		raise ValueError(msg.format(F=fh, W=x.columns[4])
		
	return x
	

def betaprod(fh):
	dtype_dict = {
#		'CHR': str,
		'SNP': str,
#		'CM': float,
#		'BP', int,
		'P1', float,
		'CHISQ1', float,
		'DIR1', int,
		'N1', int,
		'P2', float,
		'CHISQ2', float,
		'DIR2', int,
		'N2', int,
		'INFO1', float,
		'INFO2', float
		'MAF1', float,
		'MAF2', float
	}
	usecols = dtype_dict.keys()
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict)
	except AttributeError as e:
		raise AttributeError('Improperly formatted betaprod file: '+ e)
		
	if x.columns[1] != 'SNP':
		raise	ValueError('Improperly formatted betaprod file, first column should be SNP.')

	check_rsid(x['SNP'])
	for i in ['1','2']:
		N='N'+i; P='P'+i; CHISQ='CHISQ'+i; DIR='DIR'+i; MAF='MAF'+i; INFO='INFO'+i
		if N not in x.columns:
			raise ValueError('No column named {C} in betaprod.'.format(C=N))
		if DIR not in x.columns:
			raise ValueError('No column named {C} in betaprod.'.format(C=DIR))
		check_N(x[N])
		check_dir(x[DIR])
		if CHISQ in x.columns:
			check_chisq(x[CHISQ])
			betahat = np.sqrt[CHISQ] * dir
			x[CHISQ] = betahat
			ii = x.columns == CHISQ
			x.columns[ii] = 'BETAHAT'+i
		elif P in x.columns:
			check_pvalue(x[P])
			betahat = np.sqrt(chdtri(1, x[P])) * dir
			x[P] = betahat
			ii = x.columns == P
			x.columns[ii] = 'BETAHAT'+i
		else:
			raise ValueError('No column named P{i} or CHISQ{i} in betaprod.'.format(i=i))

		if MAF in x.columns:
			check_maf(x[MAF])
			x[MAF] = x['MAF'] = np.min(x['MAF'], 1-x['MAF'])
		
	return x

	
def ldscore(fh):
	fname = fh + '.l2.ldscore'
	x = pd.read_csv(fname, header=0, delim_whitespace=True)
	x.drop(['CHR','BP','CM','MAF'],axis=1)
	check_rsid(x['SNP']) 
	x.ix[1:len(x.columns)] = x.ix[1:len(x.columns)].astype(float)
	return x


def ldscore22(fh):
	chr_ld = []
	for i in xrange(23):
		chr_fh = fh + '.' + str(i)
		x = pd.read_csv(fname, header=0, delim_whitespace=True)
		x.drop(['CHR','BP','CM','MAF'],axis=1)
		chr_ld.append(x)
		
	x = pd.concat(chr_ld)
	x.ix[1:len(x.columns)] = x.ix[1:len(x.columns)].astype(float)
	check_rsid(x['SNP']) # in case there are duplicated rs#'s on different chromosomes
	return x
	
	
def M(fh):
	fname = fh + '.l2.M'
	x = open(fh, 'r').readline().split()
	x = [float(y) for y in x]
	return x
	
	
def M22(fh):
	chr_M = []
	for i in xrange(23):
		chr_fh = fh + '.' + str(i)
		chr_M.append(M(chr_fh))
		
	return np.sum(chr_M, axis=0)