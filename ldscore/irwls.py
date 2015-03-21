'''
(c) 2015 Brendan Bulik-Sullivan and Hilary Finucane

Iterativey re-weighted least squares.

'''
from __future__ import division
import numpy as np
import jackknife as jk
from docshapes import docshapes


class IRWLS(object):

    '''
    Iteratively re-weighted least squares (FLWS).

    Parameters
    ----------
    x np.ndarray with shape (n, p)
        Independent variable.
    y np.ndarray with shape (n, 1)
        Dependent variable.
    update_func : function
        Transforms output of np.linalg.lstsq to new weights.
    n_blocks : int
        Number of jackknife blocks (for estimating SE via block jackknife).
    w np.ndarray with shape (n, 1)
        Initial regression weights (default is the identity matrix). These should be on the
        inverse CVF scale.
    slow : bool
        Use slow block jackknife? (Mostly for testing)

    Attributes
    ----------
    est np.ndarray with shape (1, p)
        IRWLS estimate.
    jknife_est np.ndarray with shape (1, p)
        Jackknifed estimate.
    jknife_var np.ndarray with shape (1, p)
        Variance of jackknifed estimate.
    jknife_se np.ndarray with shape (1, p)
        Standard error of jackknifed estimate, equal to sqrt(jknife_var).
    jknife_cov np.ndarray with shape (p, p)
        Covariance matrix of jackknifed estimate.
    delete_values np.ndarray with shape (n_blocks, p)
        Jackknife delete values.

    Methods
    -------
    wls(x, y, w) :
        Weighted Least Squares.
    _weight(x, w) :
        Weight x by w.

    '''
    @docshapes(init=True)
    def __init__(self, x, y, update_func, n_blocks, w=None, slow=False, separators=None):
        n, p = x.shape
        if w is None:
            w = np.ones_like(y)
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        jknife = self.irwls(
            x, y, update_func, n_blocks, w, slow=slow, separators=separators)
        self.est = jknife.est
        self.jknife_se = jknife.jknife_se
        self.jknife_est = jknife.jknife_est
        self.jknife_var = jknife.jknife_var
        self.jknife_cov = jknife.jknife_cov
        self.delete_values = jknife.delete_values
        self.separators = jknife.separators

    @classmethod
    @docshapes
    def irwls(cls, x, y, update_func, n_blocks, w, slow=False, separators=None):
        '''
        Iteratively re-weighted least squares (IRWLS).

        Parameters
        ----------
        x np.ndarray with shape (n, p)
            Independent variable.
        y np.ndarray with shape (n, 1)
            Dependent variable.
        update_func: function
            Transforms output of np.linalg.lstsq to new weights.
        n_blocks : int
            Number of jackknife blocks (for estimating SE via block jackknife).
        w np.ndarray with shape (n, 1)
            Initial regression weights.
        slow : bool
            Use slow block jackknife? (Mostly for testing)
        separators : list or None
            Block jackknife block boundaries (optional).

        Returns
        -------
        jknife : jk.LstsqJackknifeFast
            Block jackknife regression with the final IRWLS weights.

        '''
        n, p = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        w = np.sqrt(w)
        for i in xrange(2):  # update this later
            new_w = np.sqrt(update_func(cls.wls(x, y, w)))
            if new_w.shape != w.shape:
                print 'IRWLS update:', new_w.shape, w.shape
                raise ValueError('New weights must have same shape.')
            else:
                w = new_w

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        if slow:
            jknife = jk.LstsqJackknifeSlow(
                x, y, n_blocks, separators=separators)
        else:
            jknife = jk.LstsqJackknifeFast(
                x, y, n_blocks, separators=separators)

        return jknife

    @classmethod
    @docshapes
    def wls(cls, x, y, w):
        '''
        Weighted least squares.

        Parameters
        ----------
        x np.ndarray with shape (n, p)
            Independent variable.
        y np.ndarray with shape (n, 1)
            Dependent variable.
        w np.ndarray with shape (n, 1)
            Regression weights (1/CVF scale).

        Returns
        -------
        coef : list with four elements (coefficients, residuals, rank, singular values)
            Output of np.linalg.lstsq

        '''
        n, p = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        coef = np.linalg.lstsq(x, y)
        return coef

    @classmethod
    @docshapes
    def _weight(cls, x, w):
        '''
        Weight x by w.

        Parameters
        ----------
        x np.ndarray with shape (n, p)
            Rows are observations.
        w np.ndarray with shape (n, 1)
            Regression weights (1 / sqrt(CVF) scale).

        Returns
        -------
        x_new np.ndarray with shape (n, p)
            x_new[i,j] = x[i,j] * w'[i], where w' is w normalized to have sum 1.

        Raises
        ------
        ValueError :
            If any element of w is <= 0 (negative weights are not meaningful in WLS).

        '''
        if np.any(w <= 0):
            raise ValueError('Weights must be > 0')
        n, p = x.shape
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape (n, 1).'.format(S=w.shape))

        w = w / float(np.sum(w))
        x_new = np.multiply(x, w)
        return x_new
