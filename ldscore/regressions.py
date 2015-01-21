'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Estimators of heritability and genetic correlation.

'''
from __future__ import division
import numpy as np
from scipy.stats import norm, chi2
import jackknife as jk
from fwls import FWLS


def kill_brackets(x):
	'''Get rid of brackets and trailing whitespace in numpy arrays.'''
	return x.replace('[','').replace(']', '').strip()
	
def append_intercept(x):
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
	intercept = np.ones((n_row,1))
	x_new = np.concatenate((x, intercept), axis=1)
	return x_new
		
def gencov_obs_to_liab(gencov_obs, P1, P2, K1, K2):
	'''
	Converts genetic covariance on the observed scale in an ascertained sample to genetic
	covariance on the liability scale in the population

	Parameters
	----------
	gencov_obs : float	
		Genetic covariance on the observed scale in an ascertained sample.
	P1, P2 : float in (0,1)
		Prevalences of phenotypes 1,2 in the sample.
	K1, K2 : float in (0,1)
		Prevalences of phenotypes 1,2 in the population.
		
	Returns
	-------
	gencov_liab : float
		Genetic covariance between liabilities in the population.
	
	Note: if a trait is a QT, set P = K = None.

	'''
	c1 = 1; c2 = 1
	if P1 is not None and K1 is not None:
		c1 = np.sqrt(h2_obs_to_liab(1, P1, K1))	
	if P2 is not None and K2 is not None:
		c2 = np.sqrt(h2_obs_to_liab(1, P2, K2))
	
	return gencov_obs*c1*c2	

def h2_obs_to_liab(h2_obs, P, K):
	'''
	Converts heritability on the observed scale in an ascertained sample to heritability 
	on the liability scale in the population.

	Parameters
	----------
	h2_obs : float	
		Heritability on the observed scale in an ascertained sample.
	P : float in (0,1)
		Prevalence of the phenotype in the sample.
	K : float in (0,1)
		Prevalence of the phenotype in the population.
		
	Returns
	-------
	h2_liab : float
		Heritability of liability in the population.
		
	'''
	if K <= 0 or K >= 1:
		raise ValueError('K must be in the range (0,1)')
	if P <= 0 or P >= 1:
		raise ValueError('P must be in the range (0,1)')
	
	thresh = norm.isf(K)
	conversion_factor = K**2 * (1-K)**2 / (P * (1-P) * norm.pdf(thresh) **2)
	return h2_obs * conversion_factor


class LD_Score_Regression(object):

	def __init__(self, y, x, w, N, M, n_blocks, intercept=None, slow=False, twostep=False):
		M_tot = float(np.sum(M))
		n_snp, self.n_annot = x.shape
		x_tot = np.sum(x, axis=1).reshape((n_snp, 1))
		Nbar = np.mean(N) # keep condition number low
		x = np.multiply(N, x) / Nbar
		self.constrain_intercept = intercept is not None
		if not self.constrain_intercept:
			x = append_intercept(x)
			x_tot = append_intercept(x_tot)
			update_func = lambda x: self.update_func(x, x_tot, w, N, M, Nbar)
		else:
			self.intercept = intercept
			update_func = lambda x: self.update_func(x, x_tot, w, N, M, Nbar, intercept)
		
		if twostep:
			pass
		else:	
			jknife = FWLS(x, y, update_func, n_blocks, slow=slow)
			
		self.coef, self.coef_cov, self.coef_se = self._coef(jknife, Nbar)
		self.cat, self.cat_cov, self.cat_se =\
			 self._cat(jknife, M, Nbar, self.coef, self.coef_cov)
			 
		self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
		self.prop, self.prop_cov, self.prop_se =\
			 self._prop(jknife, M, Nbar, self.cat, self.tot)
			 
		self.enrichment, self.M_prop = self._enrichment(M, M_tot, self.cat, self.tot)
		self.jknife = jknife

	def _coef(self, jknife, Nbar):
		'''Get coefficient estimates + cov from the jackknife.'''
		n_annot = self.n_annot
		coef = jknife.est[0, 0:n_annot] / Nbar
		coef_cov = jknife.jknife_cov[0:n_annot, 0:n_annot] / Nbar**2
		coef_se = np.sqrt(np.diag(coef_cov))
		return coef, coef_cov, coef_se
		
	def _cat(self, jknife, M, Nbar, coef, coef_cov):
		'''Convert coefficients to per-category h2.'''
		cat = np.multiply(M, coef)
		cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
		cat_se = np.sqrt(np.diag(cat_cov))	
		return cat, cat_cov, cat_se
	
	def _tot(self, cat, cat_cov):
		'''Convert per-category h2 to total h2.'''
		tot = np.sum(cat)
		tot_cov = np.sum(cat_cov)
		tot_se = np.sqrt(tot_cov)
		return tot, tot_cov, tot_se
	
	def _prop(self, jknife, M, Nbar, cat, tot):
		'''Convert total h2 and per-category h2 to per-category proportion h2.'''
		n_annot = self.n_annot
		numer_delete_vals = np.multiply(M, jknife.delete_values[:, 0:n_annot]) / Nbar
		denom_delete_vals = np.sum(numer_delete_vals, axis=1)*np.ones(n_annot) 
		prop = jk.RatioJackknife(cat / tot, numer_delete_vals, denom_delete_vals)
		return prop.est, prop.jknife_cov, prop.jknife_se
	
	def _enrichment(self, M, M_tot, cat, tot):
		'''Compute proportion of SNPs per-category enrichment for h2.'''
		M_prop = M / M_tot
		enrichment = np.divide(cat, M) / (tot / M_tot)
		return enrichment, M_prop

	def _intercept(self, jknife):
		'''Extract intercept and intercept SE from block jackknife.'''
		n_annot = self.n_annot
		intercept = jknife.est[0, n_annot] + 1
		intercept_se = jknife.jknife_se[0, n_annot-1]
		return intercept, intercept_se

	
class Hsq(LD_Score_Regression):
		
	def __init__(self, y, x, w, N, M, n_blocks, intercept=None, slow=False, twostep=False):
		LD_Score_Regression.__init__(self, y, x, w, N, M, n_blocks, intercept, slow, twostep)
		self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
		if not self.constrain_intercept:
			self.intercept, self.intercept_se = self._intercept(self.jknife)
			self.ratio, self.ratio_se = self._ratio(self.intercept, self.intercept_se, self.mean_chisq)
		
	def update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None):
		'''
		Update function for FWLS
		
		x is the output of np.linalg.lstsq.
		x[0] is the regression coefficients
		x[0].shape is (# of dimensions, 1)
		the last element of x[0] is the intercept.

		'''
		hsq = (np.dot(M.T, x[0][0])/Nbar)[0,0]
		if intercept is None:
			intercept = x[0][1]
	
		ld = ref_ld_tot[:,0].reshape(w_ld.shape)		
		return self.weights(ld, w_ld, N, 	np.sum(M), hsq, intercept)

	def _summarize_chisq(self, chisq):
		'''Compute mean chi^2 and lambda_GC.'''
		mean_chisq = np.mean(chisq)
		# median and matrix don't play nice
		lambda_gc = np.median(np.asarray(chisq)) / 0.4549 
		return mean_chisq, lambda_gc

	def _ratio(self, intercept, intercept_se, mean_chisq):
		if mean_chisq > 1:
			ratio_se = intercept_se / (mean_chisq - 1)
			ratio = (intercept - 1) / (mean_chisq - 1)
		else:
			ratio = float('nan')
			ratio_se = float('nan')

		return ratio, ratio_se
	
	def summary(self, ref_ld_colnames):
		'''Print summary of the LD Score Regression.'''
		s = lambda x : kill_brackets(str(np.matrix(x)))
		out = ['Total observed scale h2: '+s(self.tot)+' ('+s(self.tot_se)+')']
		if self.n_annot > 1:
			out.append( 'Categories: ' + ' '.join(ref_ld_colnames))
			out.append( 'Observed scale h2: ' + s(self.cat))
			out.append( 'Observed scale h2 SE: ' + s(self.cat_se))
			out.append( 'Proportion of SNPs: '+ s(self.M_prop))
			out.append( 'Proportion of h2g: ' + s(self.prop))
			out.append( 'Enrichment: ' + s(self.enrichment))
			out.append( 'Coefficients: ' + s(self.coef))
			out.append( 'Coefficient SE: ' + s(self.coef_se))
				
		out.append( 'Lambda GC: '+ s(self.lambda_gc))
		out.append( 'Mean Chi^2: '+ s(self.mean_chisq))
		if self.constrain_intercept:
			out.append( 'Intercept: constrained to {C}'.format(C=s(self.intercept)))
		else:
			out.append( 'Intercept: '+ s(self.intercept) + ' ('+s(self.intercept_se)+')')
			if self.mean_chisq > 1:
				out.append( 'Ratio: '+s(self.ratio) + ' ('+s(self.ratio_se)+')') 
			else:
				out.append( 'Ratio: NA (mean chi^2 < 1)' )
			
		out = '\n'.join(out)	
		return kill_brackets(out)
		
	@classmethod
	def weights(self, ld, w_ld, N, M, hsq, intercept=1.0):
		'''
		Regression weights.
	
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
		hsq = max(hsq, 0.0)
		hsq = min(hsq, 1.0)
		ld = np.fmax(ld, 1.0)
		w_ld = np.fmax(w_ld, 1.0) 
		c = hsq * N / M
		het_w = 1.0 / np.square(intercept + np.multiply(c, ld))
		oc_w = 1.0 / w_ld
		w = np.multiply(het_w, oc_w)
		return w


class Gencov(LD_Score_Regression):

	def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2, n_blocks=200, intercept=None, slow=False):
		self.hsq1, self.hsq2 = hsq1, hsq2
		self.N1, self.N2 = N1, N2
		LD_Score_Regression.__init__(z1*z2, x, w, np.sqrt(N1*N2), M, n_blocks, intercept, slow)
		self.Z = self.tot / self.tot_se
		self.P_val = chi2.sf(self.Z**2, 1, loc=0, scale=1)

	def summary(self, ref_ld_colnames):
		'''Print summary of the LD Score regression.'''
		out = []
		out.append('Total observed scale gencov: '+s(self.tot)+' ('+s(self.tot_se)+')')
		out.append('Z-score: '+s(self.Z))
		out.append('P: '+s(self.P_val))		

		if self.n_annot > 1:
			out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
			out.append( 'Observed scale gencov: '+s(self.cat))
			out.append( 'Observed scale gencov SE: '+s(self.cat_se))
			out.append( 'Proportion of SNPs: '+s(self.M_prop))
			out.append( 'Proportion of gencov: ' +s(self.prop))
			out.append( 'Enrichment: '+s(self.enrichment))
		
		if self.constrain_intercept is not None:
			out.append( 'Intercept: constrained to {C}'.format(C=s(self.intercept)))
		else:
			out.append( 'Intercept: '+ s(self.intercept)+' ('+s(self.intercept_se)+')')

		out = '\n'.join(out)
		return kill_brackets(out)

	@classmethod
	def gencov_weights(ld, w_ld, N1, N2, No, M, h1, h2, rho_g, rho):
		'''
		Regression weights.
		
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

		h1 = max(h1,0) 
		h2=max(h2,0)
		h1 = min(h1,1)
		h2=min(h2,1)
		rho_g = min(rho_g,1)
		rho_g = max(rho_g, -1)	
		ld = np.fmax(ld, 1.0)
		w_ld = np.fmax(w_ld, 1.0) 
		# prevent integer division bugs with np.divide
		N1 = N1.astype(float); N2 = N2.astype(float); No = float(No)
		a = h1*ld / M + np.divide(1.0, N1)
		b = h2*ld / M + np.divide(1.0, N2)
		c = rho_g*ld / M + np.divide(No*rho, np.multiply(N1,N2))
		het_w = np.divide(1.0, np.multiply(a, b) + 2*np.square(c))
		oc_w = np.divide(1.0, w_ld)
		# the factor of 3 is for debugging -- for degenerate rg (same sumstats twice)
		# the 3 makes the h2 weights equal to the gencov weights
		w = 3*np.multiply(het_w, oc_w)
		return w
	
	
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
	n_blocks : int, default = 1000
		Number of block jackknife blocks.
	intercepts : list with length 3
		Intercepts for constrained LD Score regression. If None, then do not constrain 
		intercept. 
	
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
	def __init__(self, bhat1, bhat2, ref_ld, w_ld, N1, N2, M, intercepts, 
		N_overlap=None,	rho=None, n_blocks=200, return_silly_things=False, first_hsq=None,
		slow=False):

		self.N1 = N1
		self.N2 = N2
		self.N_overlap = N_overlap if N_overlap is not None else 0
		self.rho = rho if rho is not None else 0
		self.M = M
		self.M_tot = np.sum(M)
		self.n_annot = ref_ld.shape[1]
		self.n_snp = ref_ld.shape[0]		
		self.intercepts = intercepts
		self.huge_se_flag = False
		self.negative_hsq_flag = False
		self.out_of_bounds_flag = False
		self.tiny_hsq_flag = False
		self.return_silly_things = return_silly_things
		chisq1 = np.multiply(N1, np.square(bhat1))
		chisq2 = np.multiply(N2, np.square(bhat2))
		
		if first_hsq is None:
			self.hsq1 = Hsq(chisq1, ref_ld, w_ld, N1, M, n_blocks=n_blocks, 
				nn=False, intercept=intercepts[0], slow=slow)
		else:
			self.hsq1 = first_hsq
			
		self.hsq2 = Hsq(chisq2, ref_ld, w_ld, N2, M, n_blocks=n_blocks, nn=False, intercept=intercepts[1], slow=slow)	
		self.gencov = Gencov(bhat1, bhat2, ref_ld, w_ld, N1, N2, M, self.hsq1.tot_hsq,self.hsq2.tot_hsq, N_overlap=self.N_overlap, rho=self.rho, n_blocks=n_blocks,	intercept=intercepts[2], slow=slow)
		
		if (self.hsq1.tot_hsq <= 0 or self.hsq2.tot_hsq <= 0):
			self.negative_hsq_flag = True
			
		# total genetic correlation
		self.tot_gencor_biased = self.gencov.tot_gencov /\
			np.sqrt(self.hsq1.tot_hsq * self.hsq2.tot_hsq)
		numer_delete_values = self.cat_to_tot(self.gencov._jknife.delete_values[:,0:self.n_annot], self.M)
		hsq1_delete_values = self.cat_to_tot(self.hsq1._jknife.delete_values[:,0:self.n_annot], self.M)\
			/ self.hsq1.Nbar
		hsq2_delete_values = self.cat_to_tot(self.hsq2._jknife.delete_values[:,0:self.n_annot], self.M)\
			/ self.hsq2.Nbar
		denom_delete_values = np.sqrt(np.multiply(hsq1_delete_values, hsq2_delete_values))
		self.gencor = RatioJackknife(self.tot_gencor_biased, numer_delete_values, denom_delete_values)
		self.tot_gencor = float(self.gencor.jknife_est)
		if (self.tot_gencor > 1.2 or self.tot_gencor < -1.2):
			self.out_of_bounds_flag = True	
		elif np.isnan(self.tot_gencor):
			self.tiny_hsq_flag = True
			
		self.tot_gencor_se = float(self.gencor.jknife_se)
		if self.tot_gencor_se > 0.25:
			self.huge_se_flag	 = True
		
		self.Z = self.tot_gencor / self.tot_gencor_se
		self.P_val = chi2.sf(self.Z**2, 1, loc=0, scale=1)

	def cat_to_tot(self, x, M):
		'''Converts per-category pseudovalues to total pseudovalues.'''
		return np.dot(x, M.T)
	
	def summary(self):
		'''Print output of Gencor object.'''
		out = []
		
		if self.negative_hsq_flag and not self.return_silly_things:
			out.append('Genetic Correlation: nan (nan) (heritability estimate < 0) ')
			out.append('Z-score: nan (nan) (heritability estimate < 0)')
			out.append('P: nan (nan) (heritability estimate < 0)')
			out.append('WARNING: One of the h2 estimates was < 0. Consult the documentation.')
			out = '\n'.join(out)

		elif self.tiny_hsq_flag and not self.return_silly_things:
			out.append('Genetic Correlation: nan (nan) (heritability close to 0) ')
			out.append('Z-score: nan (nan) (heritability close to 0)')
			out.append('P: nan (nan) (heritability close to 0)')
			out.append('WARNING: one of the h2\'s was < 0 in one of the jackknife blocks. Consult the documentation.')
			out = '\n'.join(out)
		
		elif self.huge_se_flag and not self.return_silly_things:
			warn_msg = ' WARNING: asymptotic P-values may not be valid when SE(rg) is very high.'
			out.append('Genetic Correlation: '+s(self.tot_gencor)+' ('+s(self.tot_gencor_se)+')')
			out.append('Z-score: '+s(self.Z))
			out.append('P: '+s(self.P_val)+warn_msg)	
			out = '\n'.join(out)
			
		elif self.out_of_bounds_flag and not self.return_silly_things:
			out.append('Genetic Correlation: nan (nan) (rg out of bounds) ')
			out.append('Z-score: nan (nan) (rg out of bounds)')
			out.append('P: nan (nan) (rg out of bounds)')
			out.append('WARNING: rg was out of bounds. Consult the documentation.')
			out = '\n'.join(out)
			
		else:		
			out.append('Genetic Correlation: '+s(self.tot_gencor)+' ('+s(self.tot_gencor_se)+')')
			out.append('Z-score: '+s(self.Z))
			out.append('P: '+s(self.P_val))	
			if self.return_silly_things and \
				(self.huge_se_flag or self.negative_hsq_flag or self.out_of_bounds_flag or self.tiny_hsq_flag):
				out.append('WARNING: returning silly results because you asked for them.')
			out = '\n'.join(out)
			
		return kill_brackets(out)