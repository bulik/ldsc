
from ldscore.irwls import IRWLS
import unittest
import numpy as np
import nose
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises


class Test_IRWLS_2D(unittest.TestCase):

    def setUp(self):
        self.x = np.vstack([np.ones(4), [1, 4, 3, 2]]).T
        self.y = np.sum(self.x, axis=1).reshape((4, 1))
        self.w = np.abs(np.random.normal(size=4).reshape((4, 1)))
        self.w = self.w / np.sum(self.w)
        self.update_func = lambda x: np.ones((4, 1))
        print('w=\n', self.w)

    def test_weight_2d(self):
        x = np.ones((4, 2))
        assert_array_almost_equal(
            IRWLS._weight(x, self.w), np.hstack([self.w, self.w]))

    def test_wls_2d(self):
        z = IRWLS.wls(self.x, self.y, self.w)
        assert_array_almost_equal(z[0], np.ones((2, 1)))

    def test_irwls_2d(self):
        z = IRWLS.irwls(self.x, self.y, self.update_func, 2, self.w)
        assert_array_equal(z.est.shape, (1, 2))
        assert_array_almost_equal(z.est, np.ones((1, 2)))

    def test_integrates(self):
        z = IRWLS(self.x, self.y, self.update_func, 2)
        assert_array_equal(z.est.shape, (1, 2))
        assert_array_almost_equal(z.est, np.ones((1, 2)))


class Test_IRWLS_1D(unittest.TestCase):

    def setUp(self):
        self.x = np.ones((4, 1))
        self.y = np.ones((4, 1))
        self.w = np.abs(np.random.normal(size=4).reshape((4, 1)))
        self.w = self.w / np.sum(self.w)
        self.update_func = lambda x: np.ones((4, 1))
        print('w=\n', self.w)

    def test_weight_1d(self):
        assert_array_almost_equal(IRWLS._weight(self.x, self.w), self.w)

    def test_neg_weight(self):
        self.w *= 0
        assert_raises(ValueError, IRWLS._weight, self.x, self.w)

    def test_wls_1d(self):
        z = IRWLS.wls(self.x, self.y, self.w)
        assert_array_almost_equal(z[0], 1)

    def test_irwls_1d(self):
        z = IRWLS.irwls(self.x, self.y, self.update_func, 2, self.w)
        assert_array_equal(z.est.shape, (1, 1))
        assert_array_almost_equal(z.est, 1)

    def test_integrated(self):
        z = IRWLS(self.x, self.y, self.update_func, 2)
        assert_array_equal(z.est.shape, (1, 1))
        assert_array_almost_equal(z.est, 1)
