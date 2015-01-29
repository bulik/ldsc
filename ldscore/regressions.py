'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Estimators of heritability and genetic correlation.

Shape convention is (n_snp, n_annot) for all classes.
Last column = intercept.

'''
from __future__ import division
import numpy as np
from scipy.stats import norm, chi2
import jackknife as jk
from irwls import IRWLS
np.seterr(divide='raise', invalid='raise')

s = lambda x : remove_brackets(str(np.matrix(x)))

def p_z_norm(est, se):
	'''Convert estimate and se to Z-score and P-value.'''
	try:
		Z = est / se
	except (FloatingPointError, ZeroDivisionError):
		Z = float('inf')
	
	P = chi2.sf(Z**2, 1, loc=0, scale=1) # 0 if Z=inf
	return P, Z

def remove_brackets(x):
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
	if np.isnan(P) and np.isnan(K):
		return h2_obs
	if K <= 0 or K >= 1:
		raise ValueError('K must be in the range (0,1)')
	if P <= 0 or P >= 1:
		raise ValueError('P must be in the range (0,1)')
	
	thresh = norm.isf(K)
	conversion_factor = K**2 * (1-K)**2 / (P * (1-P) * norm.pdf(thresh) **2)
	return h2_obs * conversion_factor


class LD_Score_Regression(object):

	def __init__(self, y, x, w, N, M, n_blocks, intercept=None, slow=False, twostep=False):
		for i in [y, x, w, M, N]:
			try:
				if len(i.shape) != 2:
					raise TypeError('Arguments must be 2D arrays.')
			except AttributeError:
				raise TypeError('Arguments must be arrays.')
				
		n_snp, self.n_annot = x.shape
		if any(i.shape != (n_snp, 1) for i in [y, w, N]):
			raise ValueError('N, weights and response (z1z2 or chisq) must have shape (n_snp, 1).')
		if M.shape != (1, self.n_annot):
			raise ValueError('M must have shape (1, n_annot).')
		
		M_tot = float(np.sum(M))
		x_tot = np.sum(x, axis=1).reshape((n_snp, 1))
		self.constrain_intercept = intercept is not None
		self.intercept = intercept
		tot_agg = self.aggregate(y, x_tot, N, M_tot, intercept)
		initial_w = self._update_weights(x_tot, w, N, M_tot, tot_agg, intercept)
		Nbar = np.mean(N) # keep condition number low
		x = np.multiply(N, x) / Nbar
		if not self.constrain_intercept:
			x, x_tot = append_intercept(x), append_intercept(x_tot)
			yp = y
		else:
			yp = y - intercept
		
		update_func = lambda a: self._update_func(a, x_tot, w, N, M_tot, Nbar, intercept)
		if twostep:
			raise NotImplementedError('Two-step estimator coming soon!')
		else:	
			jknife = IRWLS(x, yp, update_func, n_blocks, slow=slow, w=initial_w)
			
		self.coef, self.coef_cov, self.coef_se = self._coef(jknife, Nbar)
		self.cat, self.cat_cov, self.cat_se =\
			 self._cat(jknife, M, Nbar, self.coef, self.coef_cov)
			 
		self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
		self.prop, self.prop_cov, self.prop_se =\
			 self._prop(jknife, M, Nbar, self.cat, self.tot)
			 
		self.enrichment, self.M_prop = self._enrichment(M, M_tot, self.cat, self.tot)
		if not self.constrain_intercept:
			self.intercept, self.intercept_se = self._intercept(jknife)

		self.jknife = jknife
		self.tot_delete_values = self._delete_vals_tot(jknife, Nbar, M)
		if not self.constrain_intercept:
			self.intercept_delete_values = jknife.delete_values[:, self.n_annot]
	
	@classmethod
	def aggregate(cls, y, x, N, M, intercept=None):
		if intercept is None:
			intercept = cls.__null_intercept__
			
		num = M*(np.mean(y) - intercept)
		denom = np.mean(np.multiply(x, N))
		return num / denom

	def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None):
		raise NotImplementedError
	
	def _delete_vals_tot(self, jknife, Nbar, M):
		n_annot = self.n_annot
		tot_delete_vals = jknife.delete_values[:, 0:n_annot] # shape (n_blocks, n_annot)
		tot_delete_vals = np.dot(tot_delete_vals, M.T) / Nbar # shape (n_blocks, 1)
		return tot_delete_vals 

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
		n_blocks = jknife.delete_values.shape[0]
		numer_delete_vals = np.multiply(M, jknife.delete_values[:, 0:n_annot]) / Nbar # (n_blocks, n_annot)
		denom_delete_vals = np.sum(numer_delete_vals, axis=1).reshape((n_blocks, 1))
		denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, n_annot)))
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
		intercept = jknife.est[0, n_annot]
		intercept_se = jknife.jknife_se[0, n_annot]
		return intercept, intercept_se

	
class Hsq(LD_Score_Regression):
	__null_intercept__ = 1

	def __init__(self, y, x, w, N, M, n_blocks=200, intercept=None, slow=False, twostep=False):
		LD_Score_Regression.__init__(self, y, x, w, N, M, n_blocks, intercept, slow, twostep)
		self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
		if not self.constrain_intercept:
			self.ratio, self.ratio_se = self._ratio(self.intercept, self.intercept_se, self.mean_chisq)
				
	def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None):
		'''
		Update function for IRWLS
		
		x is the output of np.linalg.lstsq.
		x[0] is the regression coefficients
		x[0].shape is (# of dimensions, 1)
		the last element of x[0] is the intercept.

		'''
		hsq = M*x[0][0]/Nbar
		if intercept is None:
			intercept = x[0][1]
	
		ld = ref_ld_tot[:,0].reshape(w_ld.shape) # remove intercept
		w = self.weights(ld, w_ld, N, M, hsq, intercept)
		return w

	def _summarize_chisq(self, chisq):
		'''Compute mean chi^2 and lambda_GC.'''
		mean_chisq = np.mean(chisq)
		# median and matrix don't play nice
		lambda_gc = np.median(np.asarray(chisq)) / 0.4549 
		return mean_chisq, lambda_gc

	def _ratio(self, intercept, intercept_se, mean_chisq):
		'''Compute ratio (intercept - 1) / (mean chi^2 -1 ).'''
		if mean_chisq > 1:
			ratio_se = intercept_se / (mean_chisq - 1)
			ratio = (intercept - 1) / (mean_chisq - 1)
		else:
			ratio = float('nan')
			ratio_se = float('nan')

		return ratio, ratio_se
	
	def summary(self, ref_ld_colnames=None, P=None, K=None):
		'''Print summary of the LD Score Regression.'''
		if P is not None and K is not None:
			T = 'Liability'
			c = h2_obs_to_liab(1, P, K)
		else:
			T = 'Observed'
			c = 1
			
		out = ['Total '+T+' scale h2: '+s(c*self.tot)+' ('+s(c*self.tot_se)+')']
		if self.n_annot > 1:
			if ref_ld_colnames is None:
				ref_ld_colnames = ['CAT_'+str(i) for i in xrange(self.n_annot)]

			out.append( 'Categories: ' + ' '.join(ref_ld_colnames))
			out.append( T+' scale h2: ' + s(c*self.cat))
			out.append( T+' scale h2 SE: ' + s(c*self.cat_se))
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
			
		return remove_brackets('\n'.join(out))
		
	def _update_weights(self, ld, w_ld, N, M, hsq, intercept):
		if intercept is None:
			intercept = self.__null_intercept__
			
		return self.weights(ld, w_ld, N, M, hsq, intercept)
		
	@classmethod
	def weights(cls, ld, w_ld, N, M, hsq, intercept=None):
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
		M : float > 0
			Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
			the regression).
		hsq : float in [0,1]
			Heritability estimate.
	
		Returns
		-------
		w : np.matrix with shape (n_snp, 1)
			Regression weights. Approx equal to reciprocal of conditional variance function.
	
		'''
		M = float(M)
		if intercept is None:
			intercept = 1
			
		hsq = max(hsq, 0.0)
		hsq = min(hsq, 1.0)
		ld = np.fmax(ld, 1.0)
		w_ld = np.fmax(w_ld, 1.0) 
		c = hsq * N / M
		het_w = 1.0 / (2*np.square(intercept + np.multiply(c, ld)))
		oc_w = 1.0 / w_ld
		w = np.multiply(het_w, oc_w)
		return w


class Gencov(LD_Score_Regression):
	__null_intercept__ = 0
	
	def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2, intercept_hsq1, intercept_hsq2,
		n_blocks=200, intercept=None, slow=False):
		self.intercept_hsq1 = intercept_hsq1;	self.intercept_hsq2 = intercept_hsq2
		self.hsq1 = hsq1;	self.hsq2 = hsq2
		self.N1 = N1;	self.N2 = N2
		LD_Score_Regression.__init__(self, z1*z2, x, w, np.sqrt(N1*N2), M, n_blocks, intercept, slow)
		self.p, self.z = p_z_norm(self.tot, self.tot_se)
		self.mean_z1z2 = np.mean(np.multiply(z1, z2))
	
	def summary(self, ref_ld_colnames, P=None, K=None):
		'''Print summary of the LD Score regression.'''
		out = []
		if P is not None and K is not None:
			T = 'Liability'
			c = gencov_obs_to_liab(1, P[0], P[1], K[0], K[1])
		else:
			T = 'Observed'
			c = 1

		out.append('Total '+T+' scale gencov: '+s(self.tot)+' ('+s(self.tot_se)+')')
		#out.append('Z-score: '+s(self.z))
		#out.append('P: '+s(self.p))		
		if self.n_annot > 1:
			out.append( 'Categories: '+ str(' '.join(ref_ld_colnames)))
			out.append( T+' scale gencov: '+s(self.cat))
			out.append( T+' scale gencov SE: '+s(self.cat_se))
			out.append( 'Proportion of SNPs: '+s(self.M_prop))
			out.append( 'Proportion of gencov: ' +s(self.prop))
			out.append( 'Enrichment: '+s(self.enrichment))	
		
		out.append('Mean z1*z2: '+s(self.mean_z1z2))
		if self.constrain_intercept:
			out.append( 'Intercept: constrained to {C}'.format(C=s(self.intercept)))
		else:
			out.append( 'Intercept: '+ s(self.intercept)+' ('+s(self.intercept_se)+')')

		return remove_brackets('\n'.join(out))

	def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None):
		'''
		Update function for IRWLS
		x is the output of np.linalg.lstsq.
		x[0] is the regression coefficients
		x[0].shape is (# of dimensions, 1)
		the last element of x[0] is the intercept.

		'''
		rho_g = M* x[0][0]/Nbar
		if intercept is None: # i.e,. if the regression includes an intercept
			intercept = x[0][1]
	
		ld = ref_ld_tot[:,0].reshape(w_ld.shape) # remove intercept if we have one
		w = self.weights(ld, w_ld, self.N1, self.N2, np.sum(M), self.hsq1, self.hsq2, rho_g, 
			intercept, self.intercept_hsq1, self.intercept_hsq2)

		return w
	
	def _update_weights(self, ld, w_ld, sqrt_n1n2, M, rho_g, intercept):
		'''Weight function with the same signature for Hsq and Gencov.'''
		w = self.weights(ld, w_ld, self.N1, self.N2, M, self.hsq1, self.hsq2, rho_g,
			intercept, self.intercept_hsq1, self.intercept_hsq2)	
		return w
	
	@classmethod
	def weights(cls, ld, w_ld, N1, N2, M, h1, h2, rho_g, intercept_rg=None, 
		intercept_hsq1=None, intercept_hsq2=None):
		'''
		Regression weights.
		
		Parameters
		----------
		ld : np.matrix with shape (n_snp, 1) 
			LD Scores (non-partitioned)
		w_ld : np.matrix with shape (n_snp, 1)
			LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included 
			in the regression.
		M : float > 0
			Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
			the regression).
		N1, N2 :  np.matrix of ints > 0 with shape (n_snp, 1)
			Number of individuals sampled for each SNP for each study.
		h1, h2 : float in [0,1]
			Heritability estimates for each study.
		rhog : float in [0,1]
			Genetic covariance estimate.
		intercept : float 
			Genetic covariance intercept, on the z1*z2 scale (so should be Ns*rho/sqrt(N1*N2)).
	
		Returns
		-------
		w : np.matrix with shape (n_snp, 1)
			Regression weights. Approx equal to reciprocal of conditional variance function.
	
		'''
		M = float(M)
		if intercept_rg is None:
			intercept_rg = 0
		if intercept_hsq1 is None:	
			intercept_hsq1 = 1
		if intercept_hsq2 is None:
			intercept_hsq2 = 1
			
		h1 = max(h1, 0.0); h2 = max(h2, 0.0)
		h1 = min(h1, 1.0); h2 = min(h2, 1.0); 
		rho_g = min(rho_g, 1.0); rho_g = max(rho_g, -1.0)	
		ld = np.fmax(ld, 1.0)
		w_ld = np.fmax(w_ld, 1.0) 
		a = np.multiply(N1, h1*ld) / M + intercept_hsq1
		b = np.multiply(N2, h2*ld) / M + intercept_hsq2
		sqrt_n1n2 = np.sqrt(np.multiply(N1, N2))
		c = np.multiply(sqrt_n1n2, rho_g*ld) / M + intercept_rg
		try:
			het_w = 1.0 / (np.multiply(a, b) + np.square(c))
		except FloatingPointError:
			# bizarre error; should never happen in practice
			raise FloatingPointError('Why did you set hsq intercept <= 0?')

		oc_w = 1.0 / w_ld
		w = np.multiply(het_w, oc_w) 
		return w
	
	
class RG(object):

	def __init__(self, z1, z2, x, w, N1, N2, M, intercept_hsq1=None, intercept_hsq2=None, 
		intercept_gencov=None, n_blocks=200, slow=False):

		self._negative_hsq = None
		n_snp, n_annot = x.shape
		hsq1 = Hsq(np.square(z1), x, w, N1, M, n_blocks=n_blocks, intercept=intercept_hsq1, slow=slow)
		hsq2 = Hsq(np.square(z2), x, w, N1, M, n_blocks=n_blocks, intercept=intercept_hsq2, slow=slow)
		gencov = Gencov(z1, z2, x, w, N1, N2, M, hsq1.tot, hsq2.tot, hsq1.intercept,
			hsq2.intercept, n_blocks,	intercept=intercept_gencov, slow=slow)
		self.hsq1, self.hsq2, self.gencov = hsq1, hsq2, gencov
		if (hsq1.tot <= 0 or hsq2.tot <= 0):
			self._negative_hsq = True
			self.rg_ratio = float('nan'); self.rg = float('nan'); self.rg_se = float('nan')
			self.p = float('nan'); self.z = float('nan')
		else:			
			rg_ratio = np.array(gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)).reshape((1,1))
			denom_delete_values = np.sqrt(np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values))
			rg = jk.RatioJackknife(rg_ratio, gencov.tot_delete_values, denom_delete_values)
			self.rg_jknife = float(rg.jknife_est)
			self.rg_se = float(rg.jknife_se)
			self.rg_ratio = float(rg_ratio)
			self.p, self.z = p_z_norm(self.rg_ratio, self.rg_se)
		
	def summary(self, silly=False):
		'''Print output of Gencor object.'''
		out = []
		if self._negative_hsq:
			out.append('Genetic Correlation: nan (nan) (h2  out of bounds) ')
			out.append('Z-score: nan (nan) (h2  out of bounds)')
			out.append('P: nan (nan) (h2  out of bounds)')
			out.append('WARNING: One of the h2\'s was out of bounds.')
			out.append('This usually indicates a data-munging error or that h2 or N is low.')					
		elif (self.rg_ratio > 1.2 or self.rg_ratio < -1.2) and not silly:
			out.append('Genetic Correlation: nan (nan) (rg out of bounds) ')
			out.append('Z-score: nan (nan) (rg out of bounds)')
			out.append('P: nan (nan) (rg out of bounds)')
			out.append('WARNING: rg was out of bounds.')
			out.append('This usually means that h2 is not significantly different from zero.')
		else:		
			out.append('Genetic Correlation: '+s(self.rg_ratio)+' ('+s(self.rg_se)+')')
			out.append('Z-score: '+s(self.z))
			out.append('P: '+s(self.p))	
			
		return remove_brackets('\n'.join(out))