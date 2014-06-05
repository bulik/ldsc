'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module contains basic functions for estimating 
	1. heritability / partitioned heritability
	2. genetic covariance
	3. genetic correlation
	4. block jackknife standard errors (hence the module name) for all of the above.
	
Numpy does this annoying thing where it treats an array of shape (M, ) very differently
from an array of shape (M, 1). In order to deal with univariate LD Score regression
and partitioned LD Score regression with the same code, everything in this module deals
with numpy matrices. 

Weird bugs may result if you pass numpy arrays or pandas dataframes without first 
converting to a matrix with the correct shape.

Terminology note -- the intended use case for this module is an LD Score regression with
n_snps SNPs included in the regression and n_annot (partitioned) LD Scores (which means 
that the number of parameters estimated in the regression will be n_annot + 1, since we
also estimate an intercept term).

'''

from __future__ import division
import numpy as np
from scipy.stats import norm


def _weight(x, w):
	
	'''
	Re-weights x by multiplying by w.
	
	Parameters
	----------
	x : np.matrix with shape (n_row, n_col)
		Rows are observations.
	w : np.matrix with shape (n_row, 1)
		Regression weights.

	Returns
	-------
	
	x_new : np.matrix with shape (n_row, n_col)
		x_new[i,j] = x[i,j] * w[i]
	
	'''
	if np.any(w <= 0):
		raise ValueError('Weights must be > 0')

	w = np.sqrt(w/ float(np.sum(w)))
	x_new = np.multiply(x, w)
	return x_new
	

def _append_intercept(x):

	'''
	Appends an intercept term to the design matrix for a linear regression.
	
	Parameters
	----------
	x : np.matrix with shape (n_row, n_col)
		Design matrix. Columns are predictors; rows are observations. 

	Returns
	-------
	
	x_new : np.matrix with shape (n_row, n_col+1)
		Design matrix with intercept term appended.
	
	'''
	
	n_row = x.shape[0]
	int = np.matrix(np.ones(n_row)).reshape((n_row,1))
	x_new = np.concatenate((x, int), axis=1)
	return x_new

	
def _gencov_weights(ld, w_ld, N1, N2, No, M, h1, h2, rho_g, rho):

	'''
	Computes appropriate regression weights to correct for heteroskedasticity in the 
	bivariate LDScore regression under and infinitesimal model. These regression weights are 
	approximately equal to the reciprocal of the conditional variance function
	1 / var(betahat1*betahat2 | LD Score)
	
	Parameters
	----------
	ld : np.matrix with shape (n_snp, 1) 
		LD Scores (non-partitioned)
	w_ld : np.matrix with shape (n_snp, 1)
		LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
		in the regression.
	M : int > 0
		Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
		the regression).
	N1, N2 :  np.matrix of ints > 0 with shape (n_snp, 1)
		Number of individuals sampled for each SNP for each study.
	No : int
		Number of overlapping individuals.
	h1, h2 : float in [0,1]
		Heritability estimates for each study.
	rhog : float in [0,1]
		Genetic covariance estimate.
	rho : float in [0,1]
		Phenotypic correlation estimate.
	
	Returns
	-------
	w : np.matrix with shape (n_snp, 1)
		Regression weights. Approx equal to reciprocal of conditional variance function.
	
	'''
	h1 = min(h1,1)
	h2 = min(h2,1) #negative weights cause an error
	ld = np.fmax(ld, 1.0)
	w_ld = np.fmax(w_ld, 1.0) 
	# prevent integer division bugs with np.divide
	N1 = N1.astype(float); N2 = N2.astype(float); No = float(No)
	a = h1*ld / M + np.divide((1.0-h1), N1)
	b = h2*ld / M + np.divide((1.0-h2), N2)
	c = rho_g*ld / M + np.divide(No*rho, np.multiply(N1,N2))
	het_w = np.divide(1, np.multiply(a, b) + 2*np.square(c))
	oc_w = np.divide(1.0, w_ld)
	w = np.multiply(het_w, oc_w)
	return w


def _hsq_weights(ld, w_ld, N, M, hsq):

	'''
	Computes appropriate regression weights to correct for heteroskedasticity in the LD 
	Score regression under an infinitesimal model. These regression weights are 
	approximately equal to the reciprocal of the conditional variance function
	1 / var(chi^2 | LD Score)
	
	Parameters
	----------
	ld : np.matrix with shape (n_snp, 1) 
		LD Scores (non-partitioned). 
	w_ld : np.matrix with shape (n_snp, 1)
		LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
		in the regression.
	N :  np.matrix of ints > 0 with shape (n_snp, 1)
		Number of individuals sampled for each SNP.
	M : int > 0
		Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
		the regression).
	hsq : float in [0,1]
		Heritability estimate.
	
	Returns
	-------
	w : np.matrix with shape (n_snp, 1)
		Regression weights. Approx equal to reciprocal of conditional variance function.
	
	'''
	
	ld = np.fmax(ld, 1.0)
	w_ld = np.fmax(w_ld, 1.0) 
	c = hsq * N / M
	het_w = np.divide(1.0, np.square(1.0+np.multiply(c, ld)))
	oc_w = np.divide(1.0, w_ld)
	w = np.multiply(het_w, oc_w)
	return w


def obs_to_liab(h2_obs, P, K):

	'''
	Converts heritability on the observed scale in an ascertained sample to heritability 
	on the liability scale in the population.

	Parameters
	----------
	h2_obs : float	
		Heritability on the observed scale in an ascertained sample.
	P : float in [0,1]
		Prevalence of the phenotype in the sample.
	K : float in [0,1]
		Prevalence of the phenotype in the population.
		
	Returns
	-------
	h2_liab : float
		Heritability of liability in the population and standard error.
		
	'''
	
	if K <= 0 or K >= 1:
		raise ValueError('K must be in the range (0,1)')
	if P <= 0 or P >= 1:
		raise ValueError('P must be in the range (0,1)')
	
	thresh = norm.isf(K)
	z = norm.pdf(thresh)
	conversion_factor = K**2 * (1-K)**2 / (P * (1-P) * z**2)
	h2_liab = h2_obs * conversion_factor
	return h2_liab
	

class Hsq(object):

	'''
	The conflict between python and genetics capitalization conventions (capitalize 
	objects, but additive heritability is lowercase) makes me sad :-(
	
	Class for estimating heritability / partitioned heritability from summary statistics.
	
	Parameters
	----------
	chisq : np.matrix with shape (n_snp, 1)
		Chi-square statistics. 
	ld : np.matrix with shape (n_snp, n_annot) 
		LD Scores.
	w_ld : np.matrix with shape (n_snp, 1)
		LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
		in the regression.
	N :  np.matrix of ints > 0 with shape (n_snp, 1)
		Number of individuals sampled for each SNP.
	M : np.matrix of ints with shape (1, n_annot) > 0
		Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
		the regression).
	num_blocks : int, default = 1000
		Number of block jackknife blocks.
	
	Attributes
	----------
	N : int
		Sample size. In a case/control study, this should be total sample size: number of 
		cases + number of controls. NOT some measure of effective sample size that accounts 
		for case/control ratio. The case-control ratio comes into play when converting from
		observed to liability scale (in the function obs_to_liab).
	M : np.matrix of ints with shape (1, n_annot)
		Total number of SNPs per category in the reference panel used for estimating LD Score.
	M_tot : int
		Total number of SNPs in the reference panel used for estimating LD Score.
	n_annot : int
		Number of partitions.
	n_snp : int
		Number of SNPs included in the regression.	
	mean_chisq : float
		Mean chi-square.
	w_mean_chisq : float
		Weighted mean chi-square (with the same weights as the regression).
	lambda_gc : float
		Devlin and Roeder lambda GC (median chi-square / 0.4549).
	autocor : float
		Lag-1 autocorrelation between block jackknife pseudoerrors. If much above zero, the 
		block jackknife standard error will be unreliable. This can be solved by using a 
		larger block size.
	hsq_cov : np.matrix with shape (n_annot, n_annot)
		Block jackknife estimate of variance-covariance matrix of partitioned h2 estimates.
	cat_hsq : np.matrix with shape (1, n_annot)
	 	Partitioned heritability estimates.
	cat_hsq_se : np.matrix with shape (1, n_annot)
		Standard errors of partitioned heritability estimates.
	intercept : float
		LD Score regression intercept.
	intercept_se : float
		Standard error of LD Score regression intercept.
	tot_hsq : float
		Total h2g estimate.
	tot_hsq_se : float
		Block jackknife estimate of standard error of the total h2g estimate. 
	prop_hsq : np.matrix with shape (1, n_annot)
		Proportion of h2 per annotation.
	M_prop : np.matrix with shape (1, n_annot)
		Proportion of SNPs (in reference panel) per annotation. 
	enrichment : np.matrix with shape (1, n_annot)
		Per category enrichment of h2 relative to all SNPs. Precisely, 
		(h2 per SNP in category) / (h2 per SNP overall).
	_jknife : LstsqJackknife
		Jackknife object.
		
	Methods
	-------
	_aggregate(y, x, M_tot) :
		Aggregate estimator. 
		
	'''
	
	def __init__(self, chisq, ref_ld, w_ld, N, M, num_blocks=200):
	
		self.N = N
		self.M = M
		self.M_tot = float(np.sum(M))
		self.n_annot = ref_ld.shape[1]
		self.n_snp = ref_ld.shape[0]
		self.mean_chisq = np.mean(chisq)
		# median and matrix don't play nice?
		self.lambda_gc = np.median(np.asarray(chisq)) / 0.4549 
		
		ref_ld_tot = np.sum(ref_ld, axis=1)
		agg_hsq = self._aggregate(chisq, np.multiply(N, ref_ld_tot), self.M_tot)
		weights = _hsq_weights(ref_ld_tot, w_ld, N, self.M_tot, agg_hsq) 
		if np.all(weights == 0):
			raise ValueError('Something is wrong, all regression weights are zero.')	

		x = np.multiply(N, ref_ld)
		x = _append_intercept(x)
		x = _weight(x, weights)
		y = _weight(chisq, weights)
		
		self._jknife = LstsqJackknife(x, y, num_blocks)
		self.autocor = self._jknife.autocor(1)
		no_intercept_cov = self._jknife.jknife_cov[0:self.n_annot,0:self.n_annot]
		self.hsq_cov = np.multiply(np.dot(self.M.T,self.M), no_intercept_cov)
		self.cat_hsq = np.multiply(self.M, self._jknife.est[0,0:self.n_annot])
		self.cat_hsq_se = np.multiply(self.M, self._jknife.jknife_se[0,0:self.n_annot])
		self.intercept = self._jknife.est[0,self.n_annot]
		self.intercept_se = self._jknife.jknife_se[0,self.n_annot]
		self.tot_hsq = np.sum(self.cat_hsq)
		self.tot_hsq_se = np.sqrt(np.sum(M*self._jknife.jknife_cov[0:self.n_annot,\
			0:self.n_annot]*self.M.T))
		if self.mean_chisq > 1:
			self.ratio = (self.intercept - 1) / (self.mean_chisq - 1)
		else:
			self.ratio = float('nan')
		
		self.prop_hsq = self.cat_hsq / self.tot_hsq
		self.M_prop = self.M / self.M_tot
		self.enrichment = np.divide(self.cat_hsq, self.M) / (self.tot_hsq/self.M_tot)
		
	def _aggregate(self, y, x, M_tot):
		'''Aggregate estimator. For use in regression weights.'''
		numerator = np.mean(y) - 1.0
		denominator = np.mean(x) / M_tot
		agg = numerator / denominator
		return agg


class Gencov(object):
	
	'''
	Class for estimating genetic covariance / partitioned genetic covariance from summary 
	statistics.	Inherits from Hsq, but only for _aggregate. Note: the efficiency of the 
	estimate will be improved if first you estimate heritability for each trait then 
	feed these values into hsq1 and hsq2. This is only used for the regression weights. 
	
	Could probably refactor so as to reuse more code from Hsq, but (a) the amount of 
	duplicated code is small and (b) although the procedure for estimating genetic 
	covariance and h2 is now very similar, there is no guarantee that it will stay this
	way.
	
	Parameters
	----------
	bhat1, bhat2 : np.matrix with shape (n_snp, 1)
		(Signed) effect-size estimates for each study. In a case control study, bhat should be
		the signed square root of chi-square, where the sign is + if OR > 1 and - otherwise. 
	ld : np.matrix with shape (n_snp, n_annot) 
		LD Scores.
	w_ld : np.matrix with shape (n_snp, 1)
		LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
		in the regression.
	N1, N2 :  np.matrix of ints > 0 with shape (n_snp, 1)
		Number of individuals sampled for each SNP for each study.
	M : int > 0
		Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
		the regression).
	hsq1, hsq2 : float
		Heritability estimates for each study (used in regression weights).
	N_overlap : int, default 0.
		Number of overlapping samples.
	rho : float in [-1,1]
		Estimate of total phenotypic correlation between trait 1 and trait 2. Only used for 
		regression weights, and then only when N_overlap > 0. 	
	num_blocks : int, default = 1000
		Number of block jackknife blocks.
	
	Attributes
	----------
	N1, N2 : int
		Sample sizes. In a case/control study, this should be total sample size: number of 
		cases + number of controls. NOT some measure of effective sample size that accounts 
		for case/control ratio. The case-control ratio comes into play when converting from
		observed to liability scale (in the function obs_to_liab).
	gencov_cov : np.matrix with shape (n_annot, n_annot)
		Block jackknife estimate of variance-covariance matrix of the partitioned h2 estimates.
	cat_gencov : np.matrix with shape (1, n_annot)
	 	Partitioned heritability estimates.
	cat_gencov_se : np.matrix with shape (1, n_annot)
		Standard errors of partitioned heritability estimates.
	intercept : float
		LD Score regression intercept. NB this is not on the same scale as the intercept from
		the regression chisq ~ LD Score. The intercept from the genetic covariance regression
		is on the same scale as N_overlap / (N1*N2). 
	intercept_se : float
		Standard error of LD Score regression intercept.
	tot_gencov : float
		Total h2g estimate.
	tot_gencov_se : float
		Block jackknife estimate of standard error of the total h2g estimate. 
	prop_gencov : np.matrix with shape (1, n_annot)
		Proportion of genetic covariance per annotation.
	M_prop : np.matrix with shape (1, n_annot)
		Proportion of SNPs (in reference panel) per annotation. 
	enrichment : np.matrix with shape (1, n_annot)
		Per category enrichment of h2 relative to all SNPs. Precisely, 
		(gencov per SNP in category) / (gencov per SNP overall).
	_jknife : LstsqJackknife
		Jackknife object.
		
	'''
	
	def __init__(self, bhat1, bhat2, ref_ld, w_ld, N1, N2, M, hsq1, hsq2, N_overlap=None,
		rho=None, num_blocks=200):
		
		self.N1 = N1
		self.N2 = N2
		self.N_overlap = N_overlap if N_overlap is not None else 0
		self.M = M
		self.M_tot = np.sum(M)
		self.n_annot = ref_ld.shape[1]
		self.n_snp = ref_ld.shape[0]
		
		ref_ld_tot = np.sum(ref_ld, axis=1)
		y = np.multiply(bhat1, bhat2)
		agg_gencov = self._aggregate(y, ref_ld_tot, self.M_tot)
		weights = _gencov_weights(ref_ld_tot, w_ld, N1, N2, N_overlap, self.M_tot, hsq1, hsq2, 
			agg_gencov, rho) 
		if np.all(weights == 0):
			raise ValueError('Something is wrong, all regression weights are zero.')	

		x = _append_intercept(ref_ld)
		x = _weight(x, weights)
		y = _weight(y, weights)
		
		self._jknife = LstsqJackknife(x, y, num_blocks)
		self.autocor = self._jknife.autocor(1)
		self.cat_gencov = np.multiply(self.M, self._jknife.est[0,0:self.n_annot])
		self.cat_gencov_se = np.multiply(self.M, self._jknife.jknife_se[0,0:self.n_annot])
		self.intercept = self._jknife.est[0,self.n_annot]
		self.intercept_se = self._jknife.jknife_se[0,self.n_annot]
		self.tot_gencov = np.sum(self.cat_gencov)
		self.tot_gencov_se = np.sqrt(np.sum(M*self._jknife.jknife_cov[0:self.n_annot,\
			0:self.n_annot]*self.M.T))
		
		self.prop_gencov = self.cat_gencov / self.tot_gencov
		self.M_prop = self.M / self.M_tot
		self.enrichment = np.divide(self.cat_gencov, self.M) / (self.tot_gencov/self.M_tot)

	def _aggregate(self, y, x, M_tot):
		'''Aggregate estimator. For use in regression weights.'''
		numerator = np.mean(y)
		denominator = np.mean(x) / M_tot
		agg = numerator / denominator
		return agg


class Gencor(object):

	'''
	Class for estimating genetic correlation from summary statistics. Implemented as a ratio
	estimator with block jackknife bias correction (the block jackknife allows for 
	estimation of reasonably good standard errors from dependent data and decreases the 
	bias in a ratio estimate from O(1/N) to O(1/N^2), where N = number of data points). 
	
	Parameters
	----------
	bhat1, bhat2 : np.matrix with shape (n_snp, 1)
		(Signed) effect-size estimates for each study. In a case control study, bhat should be
		the signed square root of chi-square, where the sign is + if OR > 1 and - otherwise. 
	ld : np.matrix with shape (n_snp, n_annot) 
		LD Scores.
	w_ld : np.matrix with shape (n_snp, 1)
		LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
		in the regression.
	N1, N2 :  np.matrix of ints > 0 with shape (n_snp, 1)
		Number of individuals sampled for each SNP for each study.
	M : int > 0
		Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
		the regression).
	hsq1, hsq2 : float
		Heritability estimates for each study (used in regression weights).
	N_overlap : int, default 0.
		Number of overlapping samples.
	rho : float in [-1,1]
		Estimate of total phenotypic correlation between trait 1 and trait 2. Only used for 
		regression weights, and then only when N_overlap > 0. 	
	num_blocks : int, default = 1000
		Number of block jackknife blocks.
	
	Attributes
	----------
	N1, N2 : int
		Sample sizes. In a case/control study, this should be total sample size: number of 
		cases + number of controls. NOT some measure of effective sample size that accounts 
		for case/control ratio. The case-control ratio comes into play when converting from
		observed to liability scale (in the function obs_to_liab).
	M : np.matrix of ints with shape (1, n_annot)
		Total number of SNPs per category in the reference panel used for estimating LD Score.
	M_tot : int
		Total number of SNPs in the reference panel used for estimating LD Score.
	n_annot : int
		Number of partitions.
	n_snp : int
		Number of SNPs included in the regression.	
	hsq1, hsq2 : Hsq
		Heritability estimates for traits 1 and 2, respectively.
	gencov : Gencov
		Genetic covariance estimate.
	autocor : float
		Lag-1 autocorrelation between ratio block jackknife pseudoerrors. If much above zero, 
		the block jackknife standard error will be unreliable. This can be solved by using a 
		larger block size.
	tot_gencor : float
		Total genetic correlation. 
	tot_gencor_se : float
		Genetic correlation standard error.
	_gencor : RatioJackknife
		Jackknife used for estimating genetic correlation. 

	'''
	
	def __init__(self, bhat1, bhat2, ref_ld, w_ld, N1, N2, M, N_overlap=None,	rho=None, 
		num_blocks=200):

		self.N1 = N1
		self.N2 = N2
		self.N_overlap = N_overlap if N_overlap is not None else 0
		self.rho = rho if rho is not None else 0
		self.M = M
		self.M_tot = np.sum(M)
		self.n_annot = ref_ld.shape[1]
		self.n_snp = ref_ld.shape[0]			
		
		# first hsq
		chisq1 = np.multiply(N1, np.square(bhat1))
		self.hsq1 = Hsq(chisq1, ref_ld, w_ld, N1, M, num_blocks)
		# second hsq
		chisq2 = np.multiply(N2, np.square(bhat2))
		self.hsq2 = Hsq(chisq2, ref_ld, w_ld, N2, M, num_blocks)

		# genetic covariance
		self.gencov = Gencov(bhat1, bhat2, ref_ld, w_ld, N1, N2, M, self.hsq1.tot_hsq,
			self.hsq2.tot_hsq, self.N_overlap, self.rho, num_blocks)
		
		# total genetic correlation
		self.tot_gencor_biased = self.gencov.tot_gencov /\
			np.sqrt(self.hsq1.tot_hsq * self.hsq2.tot_hsq)
		numer_delete_vals = self.cat_to_tot(self.gencov._jknife.delete_values[:,0:self.n_annot], self.M)
		hsq1_delete_vals = self.cat_to_tot(self.hsq1._jknife.delete_values[:,0:self.n_annot], self.M)
		hsq2_delete_vals = self.cat_to_tot(self.hsq2._jknife.delete_values[:,0:self.n_annot], self.M)
		denom_delete_vals = np.sqrt(np.multiply(hsq1_delete_vals, hsq2_delete_vals))
		self.gencor = RatioJackknife(self.tot_gencor_biased, numer_delete_vals, denom_delete_vals)
		self.autocor = self.gencor.autocor(1)
		self.tot_gencor = float(self.gencor.jknife_est)
		self.tot_gencor_se = float(self.gencor.jknife_se)

	def cat_to_tot(self, x, M):
		'''Converts per-category pseudovalues to total pseudovalues.'''
		return np.dot(x, M.T)
	
	
class LstsqJackknife(object):

	'''
	Least-squares block jackknife.
	
	Terminology 
	-----------
	For a delete-k block jackknife with nb blocks, define
	
	full estimate : 
		The value of the estimator applied to all the data.
	i_th block-value : 
		The value of estimator applied to the ith block of size k.
	i_th delete-k value : 
		The value of the estimator applied to the data with the ith block of size k removed.
	i_th pseudovalue :
		nb*(full estimate) - (nb - 1) * (i_th delete-k value)
	jackknife estimate:
		The mean psuedovalue (with the mean taken over all nb values of i).
	i_th psuedoerror : 
		(i_th pseudovalue) - (jackknife estimate)
	jackknife standard error / variance :
		Standard error / variance of the pseudoerrors		
	output_dim : int > 0
		Number of output parameters. Default is x.shape[1].
	
	Parameters
	----------
	x : np.matrix with shape (n_snp, n_annot)
		Predictors.
	y : np.matrix with shape (n_snp, 1)
		Response variable.
	num_blocks: int, > 0
		Number of jackknife blocks.

	Attributes
	----------
	num_blocks : int 
		Number of jackknife blocks
	est : np.matrix with shape (1, n_annot)
		Value of estimator applied to full data set.
	pseudovalues : np.matrix with shape (n_blocks, n_annot)
		Jackknife pseudovalues.
	delete_values : np.matrix
		Block jackknife delete-(block_size) values.
	jknife_est : np.matrix with shape (1, n_annot)
		Mean pseudovalue.
	jknife_var : np.matrix with shape (1, n_annot)
		Estimated variance of jackknife estimator.
	jknife_se : np.matrix with shape (1, n_annot)
		Estimated standard error of jackknife estimator.
	jknife_cov : np.matrix with shape (n_annot, n_annnot)
		Estimated variance-covariance matrix of the jackknife estimator. Diagonal entries
		are equal to jknife_var.

	Methods
	-------
	autocov(lag) : 
		Returns lag-[lag] autocovariance in jackknife pseudoerrors.
	autocor(lag) :
		Returns lag-[lag] autocorrelation (autocovariance divided by standard deviation) in 
		jackknife pseudoerrors.
	__block_vals_to_psuedovals__(block_vals, est) :
		Converts block values and full estimate to pseudovalues.
	__block_vals_to_est__() :
		Converts block values to full estimate.
		
	
	Possible TODO: impute FFT de-correlation (NP)
	
	'''

	def __init__(self, x, y, num_blocks):
		if len(x.shape) <= 1:
			x = np.atleast_2d(x).T
		if len(y.shape) <= 1:
			y = np.atleast_2d(y).T
	
		self.N = y.shape[0]
		if self.N != x.shape[0]:
			raise ValueError('Number of data points in y != number of data points in x.')
		
		self.output_dim = x.shape[1]
		self.num_blocks = num_blocks		
		if self.num_blocks > self.N:
			raise ValueError('Number of blocks > number of data points.')

		self.block_vals = self.__compute_block_vals__(x, y, num_blocks)
		self.est = self.__block_vals_to_est__(self.block_vals)
		(self.pseudovalues, self.delete_values) =\
			self.__block_vals_to_pseudovals__(self.block_vals, self.est)
		(self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
			self.__jknife__(self.pseudovalues, self.num_blocks)
			
		
	def __jknife__(self, pseudovalues, num_blocks):
		'''Converts pseudovalues to jackknife estimates'''
		jknife_est = np.mean(pseudovalues, axis=0) 
		jknife_var = np.var(pseudovalues, axis=0) / (num_blocks - 1) 
		jknife_se = np.std(pseudovalues, axis=0) / np.sqrt(num_blocks - 1)
		jknife_cov = np.cov(pseudovalues.T) / (num_blocks )
		return (jknife_est, jknife_var, jknife_se, jknife_cov)

	def __compute_block_vals__(self, x, y, num_blocks):
		N = self.N
		block_size = int(N/num_blocks)
		separators = np.arange(0,N,block_size)
		remainder = N - separators[-1]
		
		if len(separators) == num_blocks:
			separators = np.hstack([separators,N])
		else:
			for r in range(remainder):
				separators[-(r+1)] += remainder - r

		xty_block_vals = []; xtx_block_vals = []
		for i in range(len(separators)-1):	
			# s = block start SNP index; e = block end SNP index
			s = separators[i]
			e = separators[i+1]
			xty = np.dot( x[s:e,...].T, y[s:e,...] )
			xtx = np.dot( x[s:e,...].T, x[s:e,...] )
			xty_block_vals.append(xty)
			xtx_block_vals.append(xtx)
			
		block_vals = (xty_block_vals, xtx_block_vals)
		return block_vals

	def __block_vals_to_pseudovals__(self, block_vals, est):
		'''Converts block values to pseudovalues'''
		pseudovalues = np.matrix(np.zeros((self.num_blocks, self.output_dim)))
		delete_values = np.matrix(np.zeros((self.num_blocks, self.output_dim)))
		xty_blocks = block_vals[0]
		xtx_blocks = block_vals[1]
		xty_tot = np.sum(xty_blocks, axis=0)
		xtx_tot = np.sum(xtx_blocks, axis=0)
		try:
			for j in xrange(0,self.num_blocks):
				delete_xty_j = xty_tot - xty_blocks[j]
				delete_xtx_inv_j = np.linalg.inv(xtx_tot - xtx_blocks[j])
				delete_value_j = np.dot(delete_xtx_inv_j, delete_xty_j).T
				pseudovalues[j,...] = self.num_blocks*est - (self.num_blocks-1)*delete_value_j
				delete_values[j,...] = delete_value_j
		except np.linalg.linalg.LinAlgError as e:
			msg = 'Singular design matrix in at least one delete-k jackknife block. '
			msg += 'Check that you have not passed highly correlated partitioned LD Scores.'
			raise np.linalg.linalg.LinAlgError(msg, e)
			
		return (pseudovalues, delete_values)
		
	def __block_vals_to_est__(self, block_vals):
		'''Converts block values to the total (non-jackknife) estimate'''
		xty_blocks = block_vals[0]
		xtx_blocks = block_vals[1]
		xty = np.sum(xty_blocks, axis=0)
		try:
			xtx_inv = np.linalg.inv(np.sum(xtx_blocks, axis=0))
		except np.linalg.linalg.LinAlgError as e:
			msg = "Singular design matrix in at least one delete-k jackknife block. "
			msg += 'Check that you have not passed highly correlated partitioned LD Scores.'
			raise np.linalg.linalg.LinAlgError(msg, e)
	
		return np.matrix(np.dot(xtx_inv, xty).T)
		
	def autocov(self, lag):
		'''Computes lag [lag] pseudovalue autocovariance'''
		pseudoerrors = self.pseudovalues - np.mean(self.pseudovalues, axis=0)
		if lag <= 0 or lag >= self.num_blocks:
			error_msg = 'Lag >= number of blocks ({L} vs {N})'
			raise ValueError(error_msg.format(L=lag, N=self.num_blocks))

		v = pseudoerrors[lag:len(pseudoerrors),...]
		w = pseudoerrors[0:len(pseudoerrors) - lag,...]
		autocov = np.diag(np.dot(v.T, w)) / (self.num_blocks - lag)
		return autocov

	def autocor(self, lag):
	
		'''
		Computes lag [lag] pseudovalue autocorrelation. Note that this assumes that the 
		psuedovalues are generated by a stationary process, which is not quite right. 
		Nevertheless, if the lag-one autocorrelation is substantially higher than zero, it is 
		a good sign that you should consider using a larger block size in order to estimate
		the regression standard error.
		
		'''
		
		return self.autocov(lag) / np.var(self.pseudovalues, axis=0)
		

class RatioJackknife(LstsqJackknife):

	'''
	Block jackknife class for a ratio estimator (for genetic correlation estimation).
	
	Inherits from LstsqJackknife.
	
	Parameters
	----------
	est : float or np.array with shape (1, n_annot)
		(Biased) ratio estimate (e.g., if we are estimate a = b / c, est should be \
		\hat{a} = \hat{b} / \hat{c}.
	numer_delete_vals : np.matrix with shape (n_blocks, n_annot) 
		Delete-k values for the numerator.
	denom_delete_vals: np.matrix with shape (n_blocks, n_annot) 
		Delete-k values for the denominator.
		
	Warning
	-------
	If any of the delete-k block values for a category is zero, will return nan for 
	jknife_est and inf for jknife_var and jknife_se for that category. jknife_cov will
	have the expected dimension, but the row and column corresponding to covariance with
	the category with a zero delete-k block value will be nan. 
		
	'''

	def __init__(self, est, numer_delete_vals, denom_delete_vals):
		if numer_delete_vals.shape != denom_delete_vals.shape:
			raise ValueError('numer_delete_vals.shape != denom_delete_vals.shape.')

		self.est = est
		self.numer_delete_vals = numer_delete_vals 
		self.denom_delete_vals = denom_delete_vals 
		self.num_blocks = numer_delete_vals.shape[0]
		self.output_dim = numer_delete_vals.shape[1]
		self.pseudovalues = self.__delete_vals_to_pseudovals__(self.est,\
			self.denom_delete_vals, self.numer_delete_vals)
		(self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
		self.__jknife__(self.pseudovalues, self.num_blocks)
		
	def __delete_vals_to_pseudovals__(self, est, denom, numer):
		'''Converts delete-k values to pseudovalues.'''
		pseudovalues = np.matrix(np.zeros((self.num_blocks, self.output_dim)))
		for j in xrange(0,self.num_blocks):
			pseudovalues[j,...] = self.num_blocks*est - (self.num_blocks - 1)*\
				numer[j,...]/denom[j,...]

		return pseudovalues