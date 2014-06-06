from __future__ import division
import numpy as np
import scipy.stats as spstats
#import progressbar as pb

def rnorm(N, mean, var):
	'''
	Wrapper around np.random.normal and np.random.multivariate_normal that does the right 
	thing depending on the dimensions of the input covariance matrix
	
	Additionally, handles the 1D case when var == 0 correctly (by returning a vector of 
	zeros). 
	
	Parameters
	----------
	
	N : int
		Number of draws from the distribution.
	mean : ndarray of floats
		Mean of the distribution.
	var : square symmetric ndarray of floats
		Variance-covariance matrix of the distribution. The 1x1 case is handled sensibly.
	
	Returns
	-------
	
	output : ndarray of floats
		The drawn samples, with shape (N, ) if univariate or (N, # of dimensions) if
		multivariate.
	
	'''	
	# convert to matrix to make keeping track of shape easier
	var = np.matrix(var)
	mean = np.matrix(mean)

	if (var.shape == (1,1) and mean.shape == (1,1)):
		if var == 0:
			output = np.repeat(mean, N)
		else: 
			output = np.random.normal(loc=mean, scale=np.sqrt(var), size=N)
	
	elif mean.shape[1] == var.shape[0] == var.shape[1]: # MVN
		# mean must be one-dimensional --> convert back to array
		mean = np.squeeze(np.asarray(mean))
		output =  np.random.multivariate_normal(mean=mean, cov=var, size=N)
	else:
		raise ValueError('Dimensions of input not understood')

	return output


def sampleSizeMetaAnalysis(n1, n2, z1, z2):
	'''
	betahat = z score / sqrt(N)
	TODO: probably better to write a function that takes more than two vectors of 
	z-scores for meta-analyses with more than two studies
	'''
	try:
		n1 = int(n1)
		n2 = int(n2)
	except TypeError:
		raise TypeError("Could not convert n1, n2 to int")
	try:
		z1 = np.array(z1, dtype="float64")
		z2 = np.array(z2, dtype="float64")
	except TypeError:
		raise TypeError("Could not convert z1, z2 to numpy array of floats")
		
	if not len(z1) == len(z2):
		raise ValueError("z1 and z2 must have same length")
		
	n = n1 + n2
	w = np.sqrt(np.array([n1/n,n2/n]))
	
	zMeta = w[0]*z1 + w[1]*z2
	return zMeta
			

def getFST(freqs1, freqs2):
	'''
	Computes FST 
	
	WARNING: freqs should be frequency of a reference allele, not MAF
	if allele A at rs1 is at frequency 40% in p1 and 60% in p2, then 
	rs1 has the same MAF in p1 as in p2, but this is deceptive because the
	ref allele frequency is different by 20%
	'''
	try:
		freqs1 = np.array(freqs1, dtype="float64")
		freqs2 = np.array(freqs2, dtype="float64")
	except TypeError:
		raise TypeError("Could not convert arguments to numpy array of floats")
	
	c1 = np.min(freqs1) <= 0 or np.max(freqs1) >= 1
	c2 = np.min(freqs2) <= 0 or np.max(freqs2) >= 1
	if c1 or c2:
		raise ValueError("Frequencies must be in the range [0,1]")

	numerator = (freqs1 - freqs2) ** 2 
	denominator = freqs1*(1-freqs2) + freqs2*(1-freqs1)
	'''
	Estimate FST as a ratio of averages rather than average of ratios,
	ref Bhatia, ..., Price, Gen Res, 2013
	
	No correction for finite sample size, because we're assuming that these are the 
	population frequencies (which we know in simulations)
	'''
	totFST = np.mean(numerator) / np.mean(denominator)
	return totFST

			
def pointNormal(p, size=1, loc=0, scale=1):
	'''
	Samples from a point-normal distribution
	'''
	try:
		if float(p) > 1 or float(p) < 0:
			raise ValueError("p must be in the interval [0,1]") 
	except TypeError:
		raise TypeError("Count not convert p to float64")
	try:
		if int(size) < 0:
			raise ValueError("size must be greater than 0")
	except TypeError:
		raise TypeError("Could not convert size to int")
	try:
		if float(scale) <= 0:
			raise ValueError("scale must be greater than 0")
	except TypeError:
		raise TypeError("Could not convert scale to float64")
	
	output = np.zeros(size)
	nonZeros = np.random.binomial(1, p, size=size)
	numNonZeros = sum(nonZeros)
	nonZeros = np.in1d(nonZeros, np.ones(1))
	output[nonZeros] = np.random.normal(scale=scale, size=numNonZeros)
	return output + loc


def bpnorm_cov_to_cor(p1, p2, p12, var1, var2, cov):
	'''
	Convenience function that converts the parameters of a bivariate point-normal to
	correlation
	'''
	if p1<=0 or p1-p12<=0 or p2-p12<=0 or 1-p1-p2+p12<=0:
		raise ValueError('Non-positive probability')
	if abs(cov)>min(var1, var2):
		raise ValueError("cov must be in the range [-min(var1,var2),min(var1,var2)]")

	cor = p12*cov / np.sqrt(p1*var1*p2*var2)
	return cor
	

def bpnorm_cor_to_cov(p1, p2, p12, var1, var2, cor):
	'''
	Convenience function that converts the parameters of a bivariate point-normal and a 
	correlation into the covariance that gives the desired correlation
	'''
	if p1<=0 or p1-p12<=0 or p2-p12<=0 or 1-p1-p2+p12<=0:
		raise ValueError('Non-positive probability')
	
	max_cor = p12*np.sqrt(var1*var2)/(p1*var1*p2*var2)
	if abs(cor) > max_cor:
			raise ValueError("abs(cor) must be less than or equal to p12*sqrt(var1*var2)/(p1*var1+p2*var2")

	cov = cor*np.sqrt(p1*var1*p2*var2)/p12
	return cov
	

def bivariatePointNormal(p1,p2,p12,var1,var2,cov,size=1,loc=(0,0)):
	'''
	Samples from a bivariate point-normal distribution
	TODO: input checking
	
	Output shape is (size, 2)
	'''
	sd1 = np.sqrt(var1)
	sd2 = np.sqrt(var2)
	probs = [p12,p1-p12,p2-p12,1-p1-p2+p12]
	branches = np.random.choice([0,1,2,3], size=size, replace=True, p=probs)
	xynull = branches == 3
	xnull_ynorm = branches == 2
	xnorm_ynull = branches == 1
	xynorm = branches == 0
	output = np.zeros((size, 2), dtype="float64")
	output[xnorm_ynull, 0] = np.random.normal(size=np.sum(xnorm_ynull), scale=sd1)
	output[xnull_ynorm, 1] = np.random.normal(size=np.sum(xnull_ynorm), scale=sd2)
	if cov == 0:
		output[xynorm, 0] = np.random.normal(size=np.sum(xynorm), scale=sd1)
		output[xynorm, 1] = np.random.normal(size=np.sum(xynorm), scale=sd2)
	else: 
		sgn = np.sign(cov)
		cov = abs(cov)
		rx = np.random.normal(size=np.sum(xynorm), loc=0, scale=sd1)
		w = var1**2*var2/cov**2 - var1
		ry = sgn*(rx + np.random.normal(size=sum(xynorm),scale=np.sqrt(w)))*np.sqrt(var2/(var1+w))
		output[xynorm, 0] = rx
		output[xynorm, 1] = ry

	return output + loc
 

def aggregateBeta(beta, ldScore):
	'''	
	Condenses a vector of betas of length M into a vector of 
	per-LD block betas of length mEff.
	
	Parameters
	----------
	beta : np.matrix with shape (M, # phenotypes) or array with shape (M, )
		Per-normalized genotype effect size.
	ldScore : np.array with shape (M, )
		LD Scores.
		
	Returns
	-------
	output : np.array with shape (Meff, # phenotypes), where Meff = M / np.mean(ldScores)
		Per-normalized LD block genotypes.
		
	'''
	try:
		beta = np.array(beta, dtype="float64")
	except TypeError, e:
		raise TypeError("Could not convert beta to numpy array of floats")
	try:
		ldScore = np.array(ldScore, dtype="int64")
	except TypeError, e:
		raise TypeError("Could not convert ldScore to numpy array of ints")

	if len(beta) != np.sum(ldScore):
		raise ValueError("length of beta must equal sum ldScore")
	if len(beta.shape) > 1:
		num_phenos = beta.shape[1]
	else:
		num_phenos = 1
	output = np.zeros(( len(ldScore), num_phenos) )
	count = 0
	for i,l in enumerate(ldScore):
		output[i] = np.sum(beta[count:count+l,...], axis=0)
		count += l
		
	return output