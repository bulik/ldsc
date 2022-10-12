'''
(c) 2015 Brendan Bulik-Sullivan and Hilary Finucane

Iterativey re-weighted least squares.

'''

import numpy as np
from . import jackknife as jk


class IRWLS(object):

    '''
    Iteratively re-weighted least squares (FLWS).

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
        Initial regression weights (default is the identity matrix). These should be on the
        inverse CVF scale.
    slow : bool
        Use slow block jackknife? (Mostly for testing)

    Attributes
    ----------
    est : np.matrix with shape (1, p)
        IRWLS estimate.
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

    def __init__(self, x, y, update_func, n_blocks, w=None, slow=False, separators=None):
        n, p = jk._check_shape(x, y)
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
    def irwls(cls, x, y, update_func, n_blocks, w, slow=False, separators=None):
        '''
        Iteratively re-weighted least squares (IRWLS).

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
        slow : bool
            Use slow block jackknife? (Mostly for testing)
        separators : list or None
            Block jackknife block boundaries (optional).

        Returns
        -------
        jknife : jk.LstsqJackknifeFast
            Block jackknife regression with the final IRWLS weights.

        '''
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        w = np.sqrt(w)
        for i in range(2):  # update this later
            new_w = np.sqrt(update_func(cls.wls(x, y, w)))
            if new_w.shape != w.shape:
                print('IRWLS update:', new_w.shape, w.shape)
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
    def wls(cls, x, y, w):
        '''
        Weighted least squares.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        w : np.matrix with shape (n, 1)
            Regression weights (1/CVF scale).

        Returns
        -------
        coef : list with four elements (coefficients, residuals, rank, singular values)
            Output of np.linalg.lstsq

        '''
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        coef = np.linalg.lstsq(x, y, rcond=None)
        return coef

    @classmethod
    def _weight(cls, x, w):
        '''
        Weight x by w.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Rows are observations.
        w : np.matrix with shape (n, 1)
            Regression weights (1 / sqrt(CVF) scale).

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
        (n, p) = x.shape
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape (n, 1).'.format(S=w.shape))

        w = w / float(np.sum(w))
        x_new = np.multiply(x, w)
        return x_new
