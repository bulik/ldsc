
import ldscore.regressions as reg
import unittest
import numpy as np
import nose
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_equal
np.set_printoptions(precision=4)


def test_update_separators():
    ii1 = [True, True, False, True, True, False, True]
    ii2 = [True, True, False, True, True, False, False]
    ii3 = [False, True, False, True, True, False, False]
    ii4 = [False, True, False, True, True, False, True]
    ii5 = [True, True, True, True, True, True, True]
    iis = list(map(np.array, [ii1, ii2, ii3, ii4, ii5]))
    ids = np.arange(len(ii1))
    for ii in iis:
        s = np.arange(np.sum(ii) + 1)
        t = reg.update_separators(s, ii)
        assert_equal(t[0], 0)
        assert_equal(t[-1], len(ii))
        assert_array_equal(ids[ii][(s[1:-2])], ids[(t[1:-2])])


def test_p_z_norm():
    est = 10
    se = 1
    p, z = reg.p_z_norm(est, se)
    assert z == 10
    assert_array_almost_equal(p * 1e23, 1.523971)
    se = 0
    p, z = reg.p_z_norm(est, se)
    assert p == 0
    assert np.isinf(z)


def test_append_intercept():
    x = np.ones((5, 2))
    correct_x = np.ones((5, 3))
    assert_array_equal(reg.append_intercept(x), correct_x)


def test_remove_brackets():
    x = ' [] [] asdf [] '
    nose.tools.assert_equal(reg.remove_brackets(x), 'asdf')


class Test_h2_obs_to_liab(unittest.TestCase):

    def test_bad_data(self):
        assert_raises(ValueError, reg.h2_obs_to_liab, 1, 1, 0.5)
        assert_raises(ValueError, reg.h2_obs_to_liab, 1, 0.5, 1)
        assert_raises(ValueError, reg.h2_obs_to_liab, 1, 0, 0.5)
        assert_raises(ValueError, reg.h2_obs_to_liab, 1, 0.5, 0)

    def test_approx_scz(self):
        # conversion for a balanced study of a 1% phenotype is about 1/2
        x = reg.h2_obs_to_liab(1, 0.5, 0.01)
        assert_array_almost_equal(x, 0.551907298063)


class Test_gencov_obs_to_liab(unittest.TestCase):

    def test_qt(self):
        self.assertEqual(reg.gencov_obs_to_liab(1, None, None, None, None), 1)

    def test_approx_scz(self):
        x = reg.gencov_obs_to_liab(1, 0.5, None, 0.01, None)
        assert_array_almost_equal(x, np.sqrt(0.551907298063))
        x = reg.gencov_obs_to_liab(1, None, 0.5, None, 0.01)
        assert_array_almost_equal(x, np.sqrt(0.551907298063))
        x = reg.gencov_obs_to_liab(1, 0.5, 0.5, 0.01, 0.01)
        assert_array_almost_equal(x, 0.551907298063)


class Test_Hsq_1D(unittest.TestCase):

    def setUp(self):
        self.chisq = np.ones((4, 1)) * 4
        self.ld = np.ones((4, 1))
        self.w_ld = np.ones((4, 1))
        self.N = 9 * np.ones((4, 1))
        self.M = np.matrix((7))
        self.hsq = reg.Hsq(
            self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=3, intercept=1)

    def test_weights(self):
        hsq = 0.5
        w = reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, hsq)
        assert_array_equal(w.shape, self.ld.shape)
        assert_array_almost_equal(
            w[0, 0], 0.5 / np.square(1 + hsq * self.N / self.M))
        # test that it deals correctly with out-of-bounds h2
        assert_array_almost_equal(reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 1),
                                  reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 2))
        assert_array_almost_equal(reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 0),
                                  reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, -1))

    def test_summarize_chisq(self):
        chisq = np.arange(100).reshape((100, 1))
        mean_chisq, lambda_gc = self.hsq._summarize_chisq(chisq)
        assert_array_almost_equal(mean_chisq, 49.5)
        assert_array_almost_equal(lambda_gc, 108.81512420312156)

    def test_summary(self):
        # not much to test; we can at least make sure no errors at runtime
        self.hsq.summary(['asdf'])
        self.ld += np.arange(4).reshape((4, 1))
        self.chisq += np.arange(4).reshape((4, 1))
        hsq = reg.Hsq(
            self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=3)
        hsq.summary(['asdf'])
        # test ratio printout with mean chi^2 < 1
        hsq.mean_chisq = 0.5
        hsq.summary(['asdf'])

    def test_update(self):
        pass

    def test_aggregate(self):
        chisq = np.ones((10, 1)) * 3 / 2
        ld = np.ones((10, 1)) * 100
        N = np.ones((10, 1)) * 100000
        M = 1e7
        agg = reg.Hsq.aggregate(chisq, ld, N, M)
        assert_array_almost_equal(agg, 0.5)
        agg = reg.Hsq.aggregate(chisq, ld, N, M, intercept=1.5)
        assert_array_almost_equal(agg, 0)


class Test_Coef(unittest.TestCase):

    def setUp(self):
        self.hsq1 = 0.2
        self.hsq2 = 0.7
        ld = (np.abs(np.random.normal(size=800)) + 1).reshape((400, 2))
        N = np.ones((400, 1)) * 1e5
        self.M = np.ones((1, 2)) * 1e7 / 2.0
        chisq = 1 + 1e5 * (ld[:, 0] * self.hsq1 / self.M[0, 0] +
                           ld[:, 1] * self.hsq2 / self.M[0, 1]).reshape((400, 1))
        w_ld = np.ones_like(chisq)
        self.hsq_noint = reg.Hsq(
            chisq, ld, w_ld, N, self.M, n_blocks=3, intercept=1)
        self.hsq_int = reg.Hsq(chisq, ld, w_ld, N, self.M, n_blocks=3)
        print(self.hsq_noint.summary())
        print(self.hsq_int.summary())

    def test_coef(self):
        a = [self.hsq1 / self.M[0, 0], self.hsq2 / self.M[0, 1]]
        assert_array_almost_equal(self.hsq_noint.coef, a)
        assert_array_almost_equal(self.hsq_int.coef, a)

    def test_cat_hsq(self):
        a = [[self.hsq1, self.hsq2]]
        assert_array_almost_equal(self.hsq_noint.cat, a)
        assert_array_almost_equal(self.hsq_int.cat, a)

    def test_tot(self):
        a = self.hsq1 + self.hsq2
        assert_array_almost_equal(self.hsq_noint.tot, a)
        assert_array_almost_equal(self.hsq_int.tot, a)

    def test_prop_hsq(self):
        d = self.hsq1 + self.hsq2
        a = [[self.hsq1 / d, self.hsq2 / d]]
        assert_array_almost_equal(self.hsq_noint.prop, a)
        assert_array_almost_equal(self.hsq_int.prop, a)

    def test_enrichment(self):
        d = self.hsq1 + self.hsq2
        a = [[self.hsq1 / (0.5 * d), self.hsq2 / (0.5 * d)]]
        assert_array_almost_equal(self.hsq_noint.enrichment, a)
        assert_array_almost_equal(self.hsq_int.enrichment, a)

    def test_intercept(self):
        assert_array_almost_equal(self.hsq_int.intercept, 1)
        assert_array_almost_equal(self.hsq_int.ratio, 0)


class Test_Hsq_2D(unittest.TestCase):

    def setUp(self):
        self.chisq = np.ones((17, 1)) * 4
        self.ld = np.hstack(
            [np.ones((17, 1)), np.arange(17).reshape((17, 1))]).reshape((17, 2))
        self.w_ld = np.ones((17, 1))
        self.N = 9 * np.ones((17, 1))
        self.M = np.matrix((7, 2))
        self.hsq = reg.Hsq(
            self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=3, intercept=1)

    def test_summary(self):
        # not much to test; we can at least make sure no errors at runtime
        self.hsq.summary(['asdf', 'qwer'])
        # change to random 7/30/2019 to avoid inconsistent singular matrix errors
        self.ld += np.random.normal(scale=0.1, size=(17,2))
        self.chisq += np.arange(17).reshape((17, 1))
        hsq = reg.Hsq(
            self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=3)
        hsq.summary(['asdf', 'qwer'])
        # test ratio printout with mean chi^2 < 1
        hsq.mean_chisq = 0.5
        hsq.summary(['asdf', 'qwer'])


class Test_Gencov_1D(unittest.TestCase):

    def setUp(self):
        self.z1 = np.ones((4, 1)) * 4
        self.z2 = np.ones((4, 1))
        self.ld = np.ones((4, 1))
        self.w_ld = np.ones((4, 1))
        self.N1 = 9 * np.ones((4, 1))
        self.N2 = 7 * np.ones((4, 1))
        self.M = np.matrix((7))
        self.hsq1 = 0.5
        self.hsq2 = 0.6
        self.gencov = reg.Gencov(self.z1, self.z2, self.ld, self.w_ld, self.N1, self.N2,
                                 self.M, self.hsq1, self.hsq2, 1.0, 1.0, n_blocks=3, intercept_gencov=1)

    def test_weights(self):
        # check that hsq weights = gencov weights when z1 = z2
        ld = np.abs(np.random.normal(size=100)).reshape((100, 1))
        w_ld = np.abs(np.random.normal(size=100)).reshape((100, 1))
        N1 = np.abs(np.random.normal(size=100)).reshape((100, 1))
        N2 = N1
        M = 10
        h1, h2, rho_g = 0.5, 0.5, 0.5
        wg = reg.Gencov.weights(
            ld, w_ld, N1, N2, M, h1, h2, rho_g, intercept_gencov=1.0)
        wh = reg.Hsq.weights(ld, w_ld, N1, M, h1, intercept=1.0)
        assert_array_almost_equal(wg, wh)

    def test_update(self):
        pass

    def test_summary(self):
        # not much to test; we can at least make sure no errors at runtime
        self.gencov.summary(['asdf'])
        self.ld += np.arange(4).reshape((4, 1))
        self.z1 += np.arange(4).reshape((4, 1))
        gencov = reg.Gencov(self.z1, self.z2, self.ld, self.w_ld, self.N1, self.N2,
                            self.M, self.hsq1, self.hsq2, 1.0, 1.0, n_blocks=3)
        gencov.summary(['asdf'])

    def test_aggregate(self):
        z1z2 = np.ones((10, 1)) / 2
        ld = np.ones((10, 1)) * 100
        N = np.ones((10, 1)) * 100000
        M = 1e7
        agg = reg.Gencov.aggregate(z1z2, ld, N, M)
        assert_array_almost_equal(agg, 0.5)
        agg = reg.Gencov.aggregate(z1z2, ld, N, M, intercept=0.5)
        assert_array_almost_equal(agg, 0)


class Test_Gencov_2D(unittest.TestCase):

    def setUp(self):
        self.ld = np.abs(np.random.normal(size=100).reshape((50, 2))) + 2
        self.z1 = np.random.normal(size=50).reshape((50, 1))
        self.z2 = np.random.normal(size=50).reshape((50, 1))
        self.w_ld = np.random.normal(size=50).reshape((50, 1))
        self.N1 = 9 * np.ones((50, 1))
        self.N2 = 7 * np.ones((50, 1))
        self.M = np.matrix((700, 222))
        self.hsq1 = 0.5
        self.hsq2 = 0.6
        self.gencov = reg.Gencov(self.z1, self.z2, self.ld, self.w_ld, self.N1, self.N2,
                                 self.M, self.hsq1, self.hsq2, 1.0, 1.0, n_blocks=3, intercept_gencov=1)

    def test_summary(self):
        # not much to test; we can at least make sure no errors at runtime
        self.gencov.summary(['asdf', 'qwer'])

    def test_eq_hsq(self):
        '''
        Gencov should be the same as hsq if z1 = z2, hsq + intercept_hsq are 0 and
        all intermediate rg's are > 0 (because Hsq.weights lower-bounds the hsq guess at 0
        but Gencov.weights lower-bounds the rho_g guess at -1). The setup below guarantees
        that all intermediate rho_g guesses will be 1

        '''
        self.ld = np.abs(np.random.normal(size=100).reshape((50, 2))) + 2
        self.z1 = (np.sum(self.ld, axis=1) + 10).reshape((50, 1))
        gencov = reg.Gencov(self.z1, self.z1, self.ld, self.w_ld, self.N1, self.N1,
                            self.M, 0, 0, 0, 0, n_blocks=3, intercept_gencov=1)
        hsq = reg.Hsq(np.square(self.z1), self.ld, self.w_ld,
                      self.N1, self.M, n_blocks=3, intercept=1)
        print(gencov.summary(['asdf', 'asdf']))
        print()
        print(hsq.summary(['asdf', 'asdf']))
        assert_array_almost_equal(gencov.tot, hsq.tot)
        assert_array_almost_equal(gencov.tot_se, hsq.tot_se)
        assert_array_almost_equal(gencov.tot_cov, hsq.tot_cov)


class Test_RG_2D(unittest.TestCase):

    def setUp(self):
        self.ld = np.abs(np.random.normal(size=100).reshape((50, 2))) + 2
        self.z1 = (np.sum(self.ld, axis=1) * 10).reshape((50, 1))
        self.w_ld = np.random.normal(size=50).reshape((50, 1))
        self.N1 = 9 * np.ones((50, 1))
        self.N2 = 7 * np.ones((50, 1))
        self.M = np.matrix((700, 222))
        self.hsq1 = 0.5
        self.hsq2 = 0.6
        self.rg = reg.RG(self.z1, -self.z1, self.ld, self.w_ld, self.N1, self.N1,
                         self.M, 1.0, 1.0, 0, n_blocks=20)

    def test_summary(self):
        # just make sure it doesn't encounter any errors at runtime
        print(self.rg.summary())
        print(self.rg.summary(silly=True))

    def test_rg(self):
        # won't be exactly 1 because the h2 values passed to Gencov aren't 0
        assert np.abs(self.rg.rg_ratio + 1) < 0.01


class Test_RG_Bad(unittest.TestCase):

    def test_negative_h2(self):
        ld = np.arange(50).reshape((50, 1)) + 0.1
        z1 = (1 / np.sum(ld, axis=1) * 10).reshape((50, 1))
        w_ld = np.ones((50, 1))
        N1 = 9 * np.ones((50, 1))
        M = np.matrix((-700))
        rg = reg.RG(z1, -z1, ld, w_ld, N1, N1,
                    M, 1.0, 1.0, 0, n_blocks=20)
        assert rg._negative_hsq
        # check no runtime errors when _negative_hsq is True
        print(rg.summary())
        print(rg.summary(silly=True))
        assert rg.rg_ratio == 'NA'
        assert rg.rg_se == 'NA'
        assert rg.rg == 'NA'
        assert rg.p == 'NA'
        assert rg.z == 'NA'
