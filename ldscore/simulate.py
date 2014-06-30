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


def bivariatePointNormal(p1,p2,p12,var1,var2,corr,size=1):
    '''
    Returns a mean-0 bivariate point normal.
    '''

    cov = corr*np.sqrt(p1*p2)/p12
    if cov > 1:
        raise ValueError("Correlation is too high for p12.")

    probs = [p12,p1-p12,p2-p12,1-p1-p2+p12]
    branches = np.random.choice([0,1,2,3], size=size, replace=True, p=probs)
    xynull = branches == 3
    xnull_ynorm = branches == 2
    xnorm_ynull = branches == 1
    xynorm = branches == 0
    output = np.zeros((size, 2), dtype="float64")
    output[xnorm_ynull, 0] = np.random.normal(0,1,np.sum(xnorm_ynull))
    output[xnull_ynorm, 1] = np.random.normal(0,1,np.sum(xnull_ynorm))

    cov = corr*np.sqrt(p1*p2)/p12
    cov_matrix = [[1,cov],[cov,1]]
    r = np.random.multivariate_normal([0,0],cov_matrix,sum(xynorm))
    output[xynorm, 0] = r[:,0]
    output[xynorm, 1] = r[:,1]

    output[:,0] = var1**.5*(output[:,0])/np.mean(output[:,0]**2)**.5
    output[:,1] = var2**.5*(output[:,1])/np.mean(output[:,1]**2)**.5

    return output
 

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