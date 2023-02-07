
import ldscore.jackknife as jk
import unittest
import numpy as np
import nose
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises


class Test_Jackknife(unittest.TestCase):

    def test_separators(self):
        N = 20
        x = np.arange(N)
        for i in range(2, int(np.floor(N / 2))):
            s = jk.Jackknife.get_separators(N, i)
            lengths = [len(x[s[j]:s[j + 1]]) for j in range(len(s) - 2)]

        self.assertTrue(max(lengths) - min(lengths) <= 1)

    def test_jknife_1d(self):
        pseudovalues = np.atleast_2d(np.arange(10)).T
        (est, var, se, cov) = jk.Jackknife.jknife(pseudovalues)
        print(jk.Jackknife.jknife(pseudovalues))
        self.assertTrue(np.isclose(var.flat, 0.91666667))
        self.assertTrue(np.isclose(est, 4.5))
        self.assertTrue(np.isclose(cov, var))
        self.assertTrue(np.isclose(se ** 2, var))
        self.assertTrue(not np.any(np.isnan(cov)))
        assert_array_equal(cov.shape, (1, 1))
        assert_array_equal(var.shape, (1, 1))
        assert_array_equal(est.shape, (1, 1))
        assert_array_equal(se.shape, (1, 1))

    def test_jknife_2d(self):
        pseudovalues = np.vstack([np.arange(10), np.arange(10)]).T
        (est, var, se, cov) = jk.Jackknife.jknife(pseudovalues)
        assert_array_almost_equal(var, np.array([[0.91666667, 0.91666667]]))
        assert_array_almost_equal(est, np.array([[4.5, 4.5]]))
        assert_array_almost_equal(
            cov, np.matrix([[0.91666667, 0.91666667], [0.91666667, 0.91666667]]))
        assert_array_almost_equal(se ** 2, var)
        assert_array_equal(cov.shape, (2, 2))
        assert_array_equal(var.shape, (1, 2))
        assert_array_equal(est.shape, (1, 2))
        assert_array_equal(se.shape, (1, 2))

    def test_delete_to_pseudo(self):
        for dim in [1, 2]:
            est = np.ones((1, dim))
            delete_values = np.ones((20, dim))
            x = jk.Jackknife.delete_values_to_pseudovalues(delete_values, est)
            assert_array_equal(x, np.ones_like(delete_values))

        est = est.T
        nose.tools.assert_raises(
            ValueError, jk.Jackknife.delete_values_to_pseudovalues, delete_values, est)


class Test_LstsqJackknifeSlow(unittest.TestCase):

    def test_delete_values_1d(self):
        func = lambda x, y: np.atleast_2d(np.sum(x + y))
        s = [0, 5, 10]
        x = np.atleast_2d(np.arange(10)).T
        y = np.atleast_2d(np.arange(10)).T
        p = jk.LstsqJackknifeSlow.delete_values(x, y, func, s)
        # 2 blocks, 1D data
        assert_array_equal(p.shape, (2, 1))
        assert_array_almost_equal(p, [[70], [20]])

    def test_delete_values_2d_1(self):
        func = lambda x, y: np.atleast_2d(np.sum(x + y, axis=0))
        s = [0, 2, 4, 6, 8, 10]
        x = np.vstack([np.arange(10), 2 * np.arange(10)]).T
        y = np.atleast_2d(np.arange(10)).T
        p = jk.LstsqJackknifeSlow.delete_values(x, y, func, s)
        # 5 blocks, 2D data
        assert_array_equal(p.shape, (5, 2))
        correct = [[88, 132],
                   [80, 120],
                   [72, 108],
                   [64,  96],
                   [56,  84]]
        assert_array_almost_equal(p, correct)

    def test_delete_values_2d_2(self):
        func = lambda x, y: np.atleast_2d(np.sum(x + y, axis=0))
        s = [0, 5, 10]
        x = np.vstack([np.arange(10), 2 * np.arange(10), 3 * np.arange(10)]).T
        y = np.atleast_2d(np.arange(10)).T
        p = jk.LstsqJackknifeSlow.delete_values(x, y, func, s)
        # 2 blocks, 3D data
        assert_array_equal(p.shape, (2, 3))
        correct = [[70, 105, 140],
                   [20,  30,  40]]
        assert_array_almost_equal(p, correct)

    def test_lstsqjackknifeslow(self):
        x = np.atleast_2d(np.arange(10)).T
        y = np.atleast_2d(2 * np.arange(10)).T
        reg = jk.LstsqJackknifeSlow(x, y, n_blocks=10)
        regnn = jk.LstsqJackknifeSlow(x, y, n_blocks=10, nn=True)
        assert_array_almost_equal(reg.est, [[2.]])
        assert_array_almost_equal(regnn.est, [[2.]])

        # TODO add tests for the SE etc

    def test_bad_data(self):
        x = np.arange(10)
        y = 2 * np.arange(9)
        assert_raises(ValueError, jk.LstsqJackknifeSlow, x, y, 10)

    def test_too_many_blocks(self):
        x = np.arange(10)
        y = 2 * np.arange(10)
        assert_raises(ValueError, jk.LstsqJackknifeSlow, x, y, 11)


class Test_LsqtsqJackknifeFast(unittest.TestCase):

    def test_block_values_1d(self):
        x = np.arange(6).reshape((6, 1))
        y = np.arange(6).reshape((6, 1))
        s = [0, 2, 4, 6]
        xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
        assert_array_equal(xty.shape, (3, 1))
        assert_array_equal(xtx.shape, (3, 1, 1))
        correct_xty = [[1], [13], [41]]
        assert_array_almost_equal(xty, correct_xty)
        correct_xtx = [[[1]], [[13]], [[41]]]
        assert_array_almost_equal(xtx, correct_xtx)

    def test_block_values_2d(self):
        x = np.vstack([np.arange(6), 2 * np.arange(6)]).T
        y = np.arange(6).reshape((6, 1))
        s = [0, 2, 4, 6]
        xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
        assert_array_equal(xty.shape, (3, 2))
        assert_array_equal(xtx.shape, (3, 2, 2))
        correct_xty = [[1, 2], [13, 26], [41, 82]]
        assert_array_almost_equal(xty, correct_xty)
        correct_xtx = [
            [[1, 2], [2, 4]],
            [[13, 26], [26, 52]],
            [[41, 82], [82, 164]]]
        assert_array_almost_equal(xtx, correct_xtx)

    def test_block_to_est_1d(self):
        x = np.arange(6).reshape((6, 1))
        y = np.arange(6).reshape((6, 1))
        for s in [[0, 3, 6], [0, 2, 4, 6], [0, 1, 5, 6]]:
            xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
            est = jk.LstsqJackknifeFast.block_values_to_est(xty, xtx)
            assert_array_equal(est.shape, (1, 1))
            assert_array_almost_equal(est, [[1.]])

    def test_block_to_est_2d(self):
        x = np.vstack([np.arange(6), [1, 7, 6, 5, 2, 10]]).T
        y = np.atleast_2d(np.sum(x, axis=1)).T
        for s in [[0, 3, 6], [0, 2, 4, 6], [0, 1, 5, 6]]:
            xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
            est = jk.LstsqJackknifeFast.block_values_to_est(xty, xtx)
            assert_array_equal(est.shape, (1, 2))
            assert_array_almost_equal(est, [[1, 1]])

        # test the dimension checking
        assert_raises(
            ValueError, jk.LstsqJackknifeFast.block_values_to_est, xty[0:2], xtx)
        assert_raises(
            ValueError, jk.LstsqJackknifeFast.block_values_to_est, xty, xtx[:, :, 0:1])
        assert_raises(
            ValueError, jk.LstsqJackknifeFast.block_values_to_est, xty, xtx[:, :, 0])

    def test_block_to_delete_1d(self):
        x = np.arange(6).reshape((6, 1))
        y = np.arange(6).reshape((6, 1))
        for s in [[0, 3, 6], [0, 2, 4, 6], [0, 1, 5, 6]]:
            xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
            delete = jk.LstsqJackknifeFast.block_values_to_delete_values(
                xty, xtx)
            assert_array_equal(delete.shape, (len(s) - 1, 1))
            assert_array_almost_equal(delete, np.ones_like(delete))

    def test_block_to_delete_2d(self):
        x = np.vstack([np.arange(6), [1, 7, 6, 5, 2, 10]]).T
        y = np.atleast_2d(np.sum(x, axis=1)).T
        for s in [[0, 3, 6], [0, 2, 4, 6], [0, 1, 5, 6]]:
            xty, xtx = jk.LstsqJackknifeFast.block_values(x, y, s)
            delete = jk.LstsqJackknifeFast.block_values_to_delete_values(
                xty, xtx)
            assert_array_equal(delete.shape, (len(s) - 1, 2))
            assert_array_almost_equal(delete, np.ones_like(delete))

    def test_eq_slow(self):
        x = np.atleast_2d(np.random.normal(size=(100, 2)))
        y = np.atleast_2d(np.random.normal(size=(100, 1)))
        print(x.shape)
        for n_blocks in range(2, 49):
            b1 = jk.LstsqJackknifeFast(x, y, n_blocks=n_blocks).est
            b2 = jk.LstsqJackknifeSlow(x, y, n_blocks=n_blocks).est
            assert_array_almost_equal(b1, b2)

    def test_bad_data(self):
        x = np.arange(6).reshape((1, 6))
        assert_raises(ValueError, jk.LstsqJackknifeFast, x, x, n_blocks=3)
        assert_raises(ValueError, jk.LstsqJackknifeFast, x.T, x.T, n_blocks=8)
        assert_raises(
            ValueError, jk.LstsqJackknifeFast, x.T, x.T, separators=list(range(10)))


class Test_RatioJackknife(unittest.TestCase):

    def test_1d(self):
        self.numer_delete_values = np.matrix(np.arange(1, 11)).T
        self.denom_delete_values = - np.matrix(np.arange(1, 11)).T
        self.denom_delete_values[9, 0] += 1
        self.est = np.matrix(-1)
        self.n_blocks = self.numer_delete_values.shape[0]
        self.jknife = jk.RatioJackknife(
            self.est, self.numer_delete_values, self.denom_delete_values)
        self.assertEqual(self.jknife.est, self.est)
        assert_array_almost_equal(self.jknife.pseudovalues[0:9, :], -1)
        self.assertEqual(self.jknife.pseudovalues[9, :], 0)
        assert_array_almost_equal(self.jknife.jknife_est[0, 0], -0.9)
        assert_array_almost_equal(self.jknife.jknife_se, 0.1)
        assert_array_almost_equal(self.jknife.jknife_var, 0.01)
        assert_array_almost_equal(self.jknife.jknife_cov, 0.01)

    def test_divide_by_zero_1d(self):
        est = np.ones((1, 1))
        numer_delete_vals = np.ones((10, 1))
        denom_delete_vals = np.ones((10, 1))
        denom_delete_vals[9, 0] = 0
        # with warnings.catch_warnings(record=True) as w:
        #        jknife = jk.RatioJackknife(est, numer_delete_vals, denom_delete_vals)
        assert_raises(FloatingPointError, jk.RatioJackknife,
                      est, numer_delete_vals, denom_delete_vals)

    def test_2d(self):
        self.numer_delete_values = np.matrix(
            np.vstack((np.arange(1, 11), 2 * np.arange(1, 11)))).T
        x = - np.arange(1, 11)
        x[9] += 1
        self.denom_delete_values = np.vstack((x, 4 * x)).T
        self.est = np.matrix((-1, -0.5))
        self.n_blocks = self.numer_delete_values.shape[0]
        self.jknife = jk.RatioJackknife(
            self.est, self.numer_delete_values, self.denom_delete_values)
        assert_array_almost_equal(self.jknife.est, self.est)
        self.assertEqual(self.jknife.est.shape, (1, 2))
        assert_array_almost_equal(self.jknife.pseudovalues[0:9, 0], -1)
        self.assertEqual(self.jknife.pseudovalues[9, 0], 0)
        assert_array_almost_equal(self.jknife.pseudovalues[0:9, 1], -0.5)
        self.assertEqual(self.jknife.pseudovalues[9, 1], 0)
        assert_array_almost_equal(self.jknife.jknife_est, [[-0.9, -0.45]])
        assert_array_almost_equal(self.jknife.jknife_se, [[0.1, 0.05]])
        assert_array_almost_equal(self.jknife.jknife_var, [[0.01, 0.0025]])
        assert_array_almost_equal(
            self.jknife.jknife_cov, np.matrix(((0.01, 0.005), (0.005, 0.0025))))

    def test_divide_by_zero_2d(self):
        est = np.ones((1, 2))
        numer_delete_vals = np.ones((10, 2))
        denom_delete_vals = np.ones((10, 2))
        denom_delete_vals[9, 0] = 0
        assert_raises(FloatingPointError, jk.RatioJackknife,
                      est, numer_delete_vals, denom_delete_vals)
