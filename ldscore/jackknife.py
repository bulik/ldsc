'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Fast block jackknives.

'''

from __future__ import division
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import nnls


def _check_shape(x, y):
	'''Check that x and y have the correct shapes (for regression jackknives).'''
	if x.shape[0] != y.shape[0]:
		raise ValueError('Number of datapoints in x != number of datapoints in y.')
	if len(x.shape) !=2 or len(y.shape) !=2:
		raise ValueError('x and y must be 2D arrays.')
	n, p = x.shape
	return (n, p)
	
def _check_shape_block(xty_block_values, xtx_block_values):
	'''Check that xty_block_values and xtx_block_values have correct shapes.'''
	if xtx_block_values.shape[0:2] != xty_block_values.shape:
		raise ValueError('Shape of xty_block_values must equal shape of first two dimensions of xty_block_values.')
	if len(xtx_block_values.shape) < 3:
		raise ValueError('xtx_block_values must be a 3D array.')
	if xtx_block_values.shape[1] != xtx_block_values.shape[2]:
		raise ValueError('Last two axes of xtx_block_values must have same dimension.')
	
	return xtx_block_values.shape[0:2]


class Jackknife(object):
	'''
	Base class for jackknife objects. Input involves x,y, so this base class is tailored
	for statistics computed from independent and dependent variables (e.g., regressions).
	The __delete_vals_to_pseudovalues__ and __jknife__ methods will still be useful for other
	sorts of statistics, but the __init__ method will need to be overriden.
	
	Parameters
	----------
	x : np.matrix with shape (n, p)
		Independent variable.
	y : np.matrix with shape (n, 1)
		Dependent variable.
	n_blocks : int
		Number of jackknife blocks
	*args, **kwargs : 
		Arguments for inheriting jackknives.
		
	Attributes
	----------
	n_blocks : int
		Number of jackknife blocks
	p : int
		Dimensionality of the independent varianble 
	N : int
		Number of datapoints (equal to x.shape[0])
		
	
	Methods
	-------
	jknife(pseudovalues):
		Computes jackknife estimate and variance from the jackknife pseudovalues.
	delete_vals_to_pseudovalues(delete_vals, est):
		Converts delete-k values and the whole-data estimate to pseudovalues.
	get_separators():
		Returns (approximately) evenly-spaced jackknife block boundaries.
	'''
	
	def __init__(self, x, y, n_blocks, *args, **kwargs):
		if len(x.shape) <= 1:
			x = np.atleast_2d(x).T
		if len(y.shape) <= 1:
			y = np.atleast_2d(y).T
	
		self.N = y.shape[0]
		if self.N != x.shape[0]:
			raise ValueError('Number of data points in y != number of data points in x.')
		
		self.n_blocks = int(n_blocks)
		self.p = x.shape[1]
		if self.n_blocks > self.N:
			msg = 'Number of jackknife blocks ({N1}) > number of datapoints ({N2}). '
			raise ValueError(msg.format(N1=n_blocks, N2=self.N))
		
		self.__init_specific__(x, y, *args, **kwargs)
	
	def __init_specific(x, y, *args, **kwargs):
		raise NotImplementedError

	@classmethod
	def jknife(self, pseudovalues):
		'''
		Converts pseudovalues to jackknife estimate and variance.
		
		Parameters
		----------
		pseudovalues : np.matrix pf floats with shape (n_blocks, p)
		
		Returns
		-------
		jknife_est : np.matrix with shape (1, p)
			Jackknifed estimate.
		jknife_var : np.matrix with shape (1, p)
			Variance of jackknifed estimate.
		jknife_se : np.matrix with shape (1, p)
			Standard error of jackknifed estimate, equal to sqrt(jknife_var).
		jknife_cov : np.matrix with shape (p, p)
			Covariance matrix of jackknifed estimate.
			
		'''
		n_blocks = pseudovalues.shape[0]
		jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / (n_blocks))
		jknife_var = np.atleast_2d(np.diag(jknife_cov))
		jknife_se = np.atleast_2d(np.sqrt(jknife_var))
		jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))
		return (jknife_est, jknife_var, jknife_se, jknife_cov)

	@classmethod
	def delete_values_to_pseudovalues(self, delete_values, est):
		'''
		Converts whole-data estimate and delete-k values to pseudovalues.
		
		Parameters
		----------
		delete_values : np.matrix with shape (n_blocks, p)
			Delete-k values.
		est : np.matrix with shape (1, p):
			Whole-data estimate.
		
		Returns
		-------
		pseudovalues : np.matrix with shape (n_blocks, p)
			Psuedovalues.
		
		Raises
		------
		ValueError :
			If est.shape != (1, delete_values.shape[1])

		'''
		n_blocks, p = delete_values.shape
		if est.shape != (1, p):
			raise ValueError('Different number of parameters in delete_values than in est.')

		return n_blocks*est - (n_blocks-1)*delete_values		
	
	@classmethod
	def get_separators(self, N, n_blocks):
		'''Define evenly-spaced block boundaries.'''
		return np.floor(np.linspace(0, N, n_blocks+1)).astype(int)
		

class LstsqJackknifeSlow(Jackknife):
	'''
	Slow linear-regression block jackknife. This class computes delete-k values directly,
	rather than forming delete-k values from block values. Useful for testing and for
	non-negative least squares (which as far as I am aware does not admit a fast block
	jackknife algorithm).
	
	Inherits from Jackknife class.
	
	Parameters
	----------
	x : np.matrix with shape (n, p)
		Independent variable.
	y : np.matrix with shape (n, 1)
		Dependent variable.
	n_blocks : int
		Number of jackknife blocks
	nn: bool
		Non-negative least-squares?
		
	Methods
	-------
	delete_values(x, y, func, s):
		Compute delete-k values of func(x, y) the slow way, with blocks defined by s.
	
	'''
	def __init_specific__(self, x, y, nn=False, separators=None):
		if nn: # non-negative least squares
			func = lambda x,y: np.atleast_2d(nnls(x, np.array(y).T[0])[0])
		else: 
			func = lambda x,y: np.atleast_2d(np.linalg.lstsq(x, np.array(y).T[0])[0])
			
		if separators is None:
			separators = self.get_separators(self.N, self.n_blocks)
			
		self.est = func(x, y)
		self.delete_values = self.delete_values(x, y, func, separators)
		self.pseudovalues = self.delete_values_to_pseudovalues(self.delete_values,self.est)
		(self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
			self.jknife(self.pseudovalues)
	
	@classmethod
	def delete_values(self, x, y, func, s):
		'''
		Compute delete values by deleting one block at a time.
		
		Parameters
		----------
		x : np.matrix with shape (n, p)
			Independent variable.
		y : np.matrix with shape (n, 1)
			Dependent variable.
		func : function (n, p) , (n, 1) --> (1, p)
			Function of x and y to be jackknived.
		s : list of ints
			Block separators.
		
		Returns
		-------
		delete_values : np.matrix with shape (n_blocks, p)
			Delete-k block values (with n_blocks blocks defined by parameter s).
		
		Raises
		------
		ValueError :
			If x.shape[0] does not equal y.shape[0] or x and y are not 2D.

		'''
		_check_shape(x, y)
		d = [func(np.vstack([x[0:s[i],...], x[s[i+1]:,...]]), np.vstack([y[0:s[i],...], y[s[i+1]:,...]]))
			 for i in xrange(len(s)-1)]
		
		return np.concatenate(d, axis=0)
	
	
class LstsqJackknifeFast(Jackknife):
	'''
	Fast block jackknife for linear regression.
	
	Inherits from Jackknife class.
	
	Parameters
	----------
	x : np.matrix with shape (n, p)
		Independent variable.
	y : np.matrix with shape (n, 1)
		Dependent variable.
	n_blocks : int
		Number of jackknife blocks
	
	Methods
	-------
	block_values(x, y, n_blocks) :
		Computes block values for the regression y~x.
	block_values_to_est(block_values) :
		Computes whole-data estimate from block values.
	block_values_to_pseudovalues(block_values, est) :
		Computes pseudovalues and delete values in a single pass over the block values.

	'''
	def __init_specific__(self, x, y):
		if self.separators is None:
			separators = self.get_separators(self.n_blocks)
			
		self.block_values = self.block_values(x, y, self.n_blocks, separators)
		self.est = self.block_values_to_est(self.block_values)
		self.delete_values = self.block_values_to_delete_values(self.block_values)
		self.pseudovalues = self.delete_values_to_pseudovalues(self.delete_values, self.est)
		(self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
			self.jknife(self.pseudovalues, self.n_blocks)		
			
	@classmethod
	def block_values(self, x, y, s):
		'''
		Compute block values.
		
		Parameters
		----------
		x : np.matrix with shape (n, p)
			Independent variable.
		y : np.matrix with shape (n, 1)
			Dependent variable.
		n_blocks : int
			Number of jackknife blocks
		s : list of ints
			Block separators.
			
		Returns
		-------
		xty_block_values : np.matrix with shape (n_blocks, p)
			Block values of X^T Y.
		xtx_block_values : 3d np array with shape (n_blocks, p, p)
			Block values of X^T X.

		Raises
		------
		ValueError :
			If x.shape[0] does not equal y.shape[0] or x and y are not 2D.

		'''
		n, p = _check_shape(x, y)
		n_blocks = len(s) - 1 
		xtx_block_values = np.zeros((n_blocks, p, p))
		xty_block_values = np.zeros((n_blocks, p))
		for i in xrange(n_blocks):
			xty_block_values[i,...] = np.dot(x[s[i]:s[i+1],...].T, y[s[i]:s[i+1],...]).reshape((1,p))
			xtx_block_values[i,...] = np.dot(x[s[i]:s[i+1],...].T, x[s[i]:s[i+1],...])

		return (xty_block_values, xtx_block_values)
	
	@classmethod
	def block_values_to_est(self, xty_block_values, xtx_block_values):
		'''
		Converts block values to the whole-data linear regression estimate.
		
		Parameters
		----------
		xty_block_values : np.matrix with shape (n_blocks, p)
			Block values of X^T Y.
		xtx_block_values : 3D np.array with shape (n_blocks, p, p)
			Block values of X^T X
		
		Returns
		-------
		est : np.matrix with shape (1, p)
			Whole data estimate.		
		
		Raises
		------
		LinAlgError : 
			If design matrix is singular.
		ValueError :
			If the last two dimensions of xtx_block_values are not equal or if the first two
	    dimensions of xtx_block_values do not equal the shape of xty_block_values.
		
		'''
		n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
		xty = np.sum(xty_block_values, axis=0)
		xtx = np.sum(xtx_block_values, axis=0)
		return np.linalg.solve(xtx, xty).reshape((1, p))

	@classmethod
	def block_values_to_delete_values(self, xty_block_values, xtx_block_values):
		'''
		Converts block values to delete values.
		
		Parameters
		----------
		xty_block_values : np.matrix with shape (n_blocks, p)
			Block values of X^T Y.
		xtx_block_values : 3D np.array with shape (n_blocks, p, p)
			Block values of X^T X
		est : np.matrix with shape (1, p)
			Whole data estimate

		Returns
		-------
		delete_values : np.matrix with shape (n_blocks, p)
			Delete Values.
		
		Raises
		------
		LinAlgError :
			If delete-k design matrix is singular.
		ValueError :
			If the last two dimensions of xtx_block_values are not equal or if the first two
	    dimensions of xtx_block_values do not equal the shape of xty_block_values.
		
		'''
		n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
		delete_values = np.zeros((n_blocks, p))
		xty_tot = np.sum(xty_blocks, axis=0)
		xtx_tot = np.sum(xtx_blocks, axis=0)
		for j in xrange(n_blocks):
			delete_xty = xty_tot - xty_block_values[j]
			delete_xtx = xtx_tot - xtx_block_values[j]
			delete_values[j,...] = np.linalg.solve(delete_xtx, delete_xty).reshape((1, p))
			
		return delete_values
		

class RatioJackknife(Jackknife):

	'''
	Block jackknife class for a ratio estimator (for genetic correlation estimation).
	
	Inherits from LstsqJackknife.
	
	Parameters
	----------
	est : float or np.array with shape (1, n_annot)
		(Biased) ratio estimate (e.g., if we are estimate a = b / c, est should be \
		\hat{a} = \hat{b} / \hat{c}.
	numer_delete_values : np.matrix with shape (n_blocks, n_annot) 
		Delete-k values for the numerator.
	denom_delete_values: np.matrix with shape (n_blocks, n_annot) 
		Delete-k values for the denominator.
		
	Warning
	-------
	If any of the delete-k block values for a category is zero, will return nan for 
	jknife_est and inf for jknife_var and jknife_se for that category. jknife_cov will
	have the expected dimension, but the row and column corresponding to covariance with
	the category with a zero delete-k block value will be nan. 
		
	'''

	def __init__(self, est, numer_delete_values, denom_delete_values):
		if numer_delete_values.shape != denom_delete_values.shape:
			raise ValueError('numer_delete_values.shape != denom_delete_values.shape.')

		self.est = est
		self.numer_delete_values = numer_delete_values 
		self.denom_delete_values = denom_delete_values 
		self.n_blocks = numer_delete_values.shape[0]
		self.p = numer_delete_values.shape[1]
		self.pseudovalues = self.__delete_values_to_pseudovalues__(self.est,\
			self.denom_delete_values, self.numer_delete_values)
		(self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
		self.__jknife__(self.pseudovalues, self.n_blocks)
	
	@classmethod
	def delete_values_to_pseudovalues(self, est, denom, numer):
		'''Converts delete-k values to pseudovalues.'''
		pseudovalues = np.matrix(np.zeros((self.n_blocks, self.p)))
		for j in xrange(0,self.n_blocks):
			pseudovalues[j,...] = self.n_blocks*est - (self.n_blocks - 1)*\
				numer[j,...]/denom[j,...]

		return pseudovalues
