from __future__ import division
import numpy as np
import pandas as pd

def check_rsid(rsids):
	pass

def chisq(fh):
	x = pd.read_csv(fh, header=0, delim_whitespace=True)
	check_rsid(x['SNP']) 
	# do stuff to the summary stats
	return x
	
def betaprod(fh):
	x = pd.read_csv(fh, header=0, delim_whitespace=True)
	check_rsid(x['SNP']) 
	# do stuff to the summary stats
	return x
	
	
def ldscore(fh):
	fname = fh + '.l2.ldscore'
	x = pd.read_csv(fname, header=0, delim_whitespace=True)
	check_rsid(x['SNP']) 
	return x

def ldscore22(fh):
	chr_ld = []
	for i in xrange(23):
		chr_fh = fh + '.' + str(i)
		chr_ld.append(ldscore(chr_fh))
		
	return np.c_(chr_ld)
	
def M(fh):
	fname = fh + '.l2.M'
	pass
	
def M22(fh):
	chr_M = []
	for i in xrange(23):
		chr_fh = fh + '.' + str(i)
		chr_M.append(M(chr_fh))
		
	return np.sum(chr_M)
		
			
	
