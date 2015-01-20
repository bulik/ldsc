'''
(c) 2015 Brendan Bulik-Sullivan and Hilary Finucane

Feasible weighted least squares.

'''
from __future__ import division
import numpy as np
import jackknife as jk

class FWLS(object):
	'''
	Feasible weighted least squares (FLWS).
	
	Parameters
	----------
	x : np.matrix with shape (n, p)
		Independent variable.
	y : np.matrix with shape (n, 1)
		Dependent variable.
	update_func : function
		Transforms output of np.linalg.lstsq to new weights.
	n_blocks : int
		Number of jackknife blocks (for estimating SE via block jackknife).
	w : np.matrix with shape (n, 1)
		Initial regression weights (default is the identity matrix).
	
	Attributes
	----------
	est : np.matrix with shape (1, p)
		FWLS estimate.
	jknife_est : np.matrix with shape (1, p)
		Jackknifed estimate.
	jknife_var : np.matrix with shape (1, p)
		Variance of jackknifed estimate.
	jknife_se : np.matrix with shape (1, p)
		Standard error of jackknifed estimate, equal to sqrt(jknife_var).
	jknife_cov : np.matrix with shape (p, p)
		Covariance matrix of jackknifed estimate.
	delete_values : np.matrix with shape (n_blocks, p)
		Jackknife delete values.

	Methods
	-------
	wls(x, y, w) :
		Weighted Least Squares.
	_weight(x, w) :
		Weight x by w.
		
	'''
	def __init__(self, x, y, update_func, n_blocks, w=None):
		print x
		n, p = jk._check_shape(x, y)
		if w is None:
			w = np.ones_like(y)
		
		jknife = self.fwls(x, y, update_func, n_blocks, w)
		self.est = jknife.est
		self.jknife_se = jknife.jknife_se
		self.jknife_est = jknife.jknife_est
		self.jknife_var = jknife.jknife_var	
		self.jknife_cov = jknife.jknife_cov
		self.delete_values = jknife.delete_values
		
	@classmethod
	def fwls(self, x, y, update_func, n_blocks, w):
		'''
		Feasible weighted least squares (FWLS).
		
		Parameters
		----------
		x : np.matrix with shape (n, p)
			Independent variable.
		y : np.matrix with shape (n, 1)
			Dependent variable.
		update_func: function
			Transforms output of np.linalg.lstsq to new weights.			
		n_blocks : int
			Number of jackknife blocks (for estimating SE via block jackknife).
		w : np.matrix with shape (n, 1)
			Initial regression weights.
		Returns
		-------
		jknife : jk.LstsqJackknifeFast
			Block jackknife regression with the final FWLS weights.
	
		'''
		for i in xrange(3): # update this later
			w = np.sqrt(update_func(self.wls(x, y, w)))
		
		x, y = self._weight(x,w), self._weight(y, w)
		jknife = jk.LstsqJackknifeFast(x, y, n_blocks)
		return jknife
		
	@classmethod
	def wls(self, x, y, w):
		'''
		Weighted least squares.
		
		Parameters
		----------
		x : np.matrix with shape (n, p)
			Independent variable.
		y : np.matrix with shape (n, 1)
			Dependent variable.
		w : np.matrix with shape (n, 1)
			Regression weights.
			
		Returns
		-------
		coef : list with four elements (coefficients, residuals, rank, singular values)
			Output of np.linalg.lstsq
	
		'''
		x = self._weight(x, w)
		y = self._weight(y, w)
		coef = np.linalg.lstsq(x, y)
		return coef

	@classmethod
	def _weight(self, x, w):
		'''
		Weight x by w.
	
		Parameters
		----------
		x : np.matrix with shape (n, p)
			Rows are observations.
		w : np.matrix with shape (n, 1)
			Regression weights.

		Returns
		-------
		x_new : np.matrix with shape (n, p)
			x_new[i,j] = x[i,j] * w'[i], where w' is w normalized to have sum 1.
		
		Raises
		------
		ValueError :
			If any element of w is <= 0 (negative weights are not meaningful in WLS).
	
		'''
		if np.any(w <= 0):
			raise ValueError('Weights must be > 0')

		w = w/ float(np.sum(w))
		x_new = np.multiply(x, w)
		return x_new