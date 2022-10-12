
import ldscore.sumstats as s
import ldscore.parse as ps
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from nose.tools import *
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
from nose.plugins.attrib import attr
import os
from ldsc import parser

DIR = os.path.dirname(__file__)
N_REP = 500
s._N_CHR = 2  # having to mock 22 files is annoying


class Mock(object):
    '''
    Dumb object for mocking args and log
    '''

    def __init__(self):
        pass

    def log(self, x):
        # pass
        print(x)

log = Mock()
args = Mock()
t = lambda attr: lambda obj: getattr(obj, attr, float('nan'))


def test_check_condnum():
    x = np.ones((2, 2))
    x[1, 1] += 1e-5
    args.invert_anyway = False
    assert_raises(ValueError, s._check_ld_condnum, args, log, x)
    args.invert_anyway = True
    s._check_ld_condnum(args, log, x)  # no error


def test_check_variance():
    ld = pd.DataFrame({'SNP': ['a', 'b', 'c'],
                       'LD1': np.ones(3).astype(float),
                       'LD2': np.arange(3).astype(float)})
    ld = ld[['SNP', 'LD1', 'LD2']]
    M_annot = np.array([[1, 2]])
    M_annot, ld, novar_col = s._check_variance(log, M_annot, ld)
    assert_array_equal(M_annot.shape, (1, 1))
    assert_array_equal(M_annot, [[2]])
    assert_allclose(ld.iloc[:, 1], [0, 1, 2])
    assert_array_equal(novar_col, [True, False])


def test_align_alleles():
    beta = pd.Series(np.ones(6))
    alleles = pd.Series(['ACAC', 'TGTG', 'GTGT', 'AGCT', 'AGTC', 'TCTC'])
    beta = s._align_alleles(beta, alleles)
    assert_series_equal(beta, pd.Series([1.0, 1, 1, -1, 1, 1]))


def test_filter_bad_alleles():
    alleles = pd.Series(['ATAT', 'ATAG', 'DIID', 'ACAC'])
    bad_alleles = s._filter_alleles(alleles)
    print(bad_alleles)
    assert_series_equal(bad_alleles, pd.Series([False, False, False, True]))


def test_read_annot():
    ref_ld_chr = None
    ref_ld = os.path.join(DIR, 'annot_test/test')
    overlap_matrix, M_tot = s._read_chr_split_files(ref_ld_chr, ref_ld, log, 'annot matrix',
                                                    ps.annot, frqfile=None)
    assert_array_equal(overlap_matrix, [[1, 0, 0], [0, 2, 2], [0, 2, 2]])
    assert_array_equal(M_tot, 3)

    frqfile = os.path.join(DIR, 'annot_test/test1')
    overlap_matrix, M_tot = s._read_chr_split_files(ref_ld_chr, ref_ld, log, 'annot matrix',
                                                    ps.annot, frqfile=frqfile)
    assert_array_equal(overlap_matrix, [[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    assert_array_equal(M_tot, 2)


def test_valid_snps():
    x = {'AC', 'AG', 'CA', 'CT', 'GA', 'GT', 'TC', 'TG'}
    assert_equal(x, s.VALID_SNPS)


def test_bases():
    x = set(['A', 'T', 'G', 'C'])
    assert_equal(x, set(s.BASES))


def test_complement():
    assert_equal(s.COMPLEMENT, {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'})


def test_warn_len():
    # nothing to test except that it doesn't throw an error at runtime
    s._warn_length(log, [1])


def test_match_alleles():
    m = {'ACAC',
         'ACCA',
         'ACGT',
         'ACTG',
         'AGAG',
         'AGCT',
         'AGGA',
         'AGTC',
         'CAAC',
         'CACA',
         'CAGT',
         'CATG',
         'CTAG',
         'CTCT',
         'CTGA',
         'CTTC',
         'GAAG',
         'GACT',
         'GAGA',
         'GATC',
         'GTAC',
         'GTCA',
         'GTGT',
         'GTTG',
         'TCAG',
         'TCCT',
         'TCGA',
         'TCTC',
         'TGAC',
         'TGCA',
         'TGGT',
         'TGTG'}
    assert_equal(m, s.MATCH_ALLELES)


def test_flip_alleles():
    m = {'ACAC': False,
         'ACCA': True,
         'ACGT': True,
         'ACTG': False,
         'AGAG': False,
         'AGCT': True,
         'AGGA': True,
         'AGTC': False,
         'CAAC': True,
         'CACA': False,
         'CAGT': False,
         'CATG': True,
         'CTAG': True,
         'CTCT': False,
         'CTGA': False,
         'CTTC': True,
         'GAAG': True,
         'GACT': False,
         'GAGA': False,
         'GATC': True,
         'GTAC': True,
         'GTCA': False,
         'GTGT': False,
         'GTTG': True,
         'TCAG': False,
         'TCCT': True,
         'TCGA': True,
         'TCTC': False,
         'TGAC': False,
         'TGCA': True,
         'TGGT': True,
         'TGTG': False}
    assert_equal(m, s.FLIP_ALLELES)


def test_strand_ambiguous():
    m = {'AC': False,
         'AG': False,
         'AT': True,
         'CA': False,
         'CG': True,
         'CT': False,
         'GA': False,
         'GC': True,
         'GT': False,
         'TA': True,
         'TC': False,
         'TG': False}
    assert_equal(m, s.STRAND_AMBIGUOUS)


@attr('rg')
@attr('slow')
class Test_RG_Statistical():

    @classmethod
    def setUpClass(cls):
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/twold_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.rg = ','.join(
            (DIR + '/simulate_test/sumstats/' + str(i) for i in range(N_REP)))
        args.out = DIR + '/simulate_test/1'
        x = s.estimate_rg(args, log)
        args.intercept_gencov = ','.join(('0' for _ in range(N_REP)))
        args.intercept_h2 = ','.join(('1' for _ in range(N_REP)))
        y = s.estimate_rg(args, log)
        cls.rg = x
        cls.rg_noint = y

    def test_rg_ratio(self):
        assert_allclose(np.nanmean(list(map(t('rg_ratio'), self.rg))), 0, atol=0.02)

    def test_rg_ratio_noint(self):
        assert_allclose(
            np.nanmean(list(map(t('rg_ratio'), self.rg_noint))), 0, atol=0.02)

    def test_rg_se(self):
        assert_allclose(np.nanmean(list(map(t('rg_se'), self.rg))), np.nanstd(
            list(map(t('rg_ratio'), self.rg))), atol=0.02)

    def test_rg_se_noint(self):
        assert_allclose(np.nanmean(list(map(t('rg_se'), self.rg_noint))), np.nanstd(
            list(map(t('rg_ratio'), self.rg_noint))), atol=0.02)

    def test_gencov_tot(self):
        assert_allclose(
            np.nanmean(list(map(t('tot'), list(map(t('gencov'), self.rg))))), 0, atol=0.02)

    def test_gencov_tot_noint(self):
        assert_allclose(
            np.nanmean(list(map(t('tot'), list(map(t('gencov'), self.rg_noint))))), 0, atol=0.02)

    def test_gencov_tot_se(self):
        assert_allclose(np.nanstd(list(map(t('tot'), list(map(t('gencov'), self.rg))))), np.nanmean(
            list(map(t('tot_se'), list(map(t('gencov'), self.rg))))), atol=0.02)

    def test_gencov_tot_se_noint(self):
        assert_allclose(np.nanstd(list(map(t('tot'), list(map(t('gencov'), self.rg_noint))))), np.nanmean(
            list(map(t('tot_se'), list(map(t('gencov'), self.rg_noint))))), atol=0.02)

    def test_gencov_cat(self):
        assert_allclose(
            np.nanmean(list(map(t('cat'), list(map(t('gencov'), self.rg))))), [0, 0], atol=0.02)

    def test_gencov_cat_noint(self):
        assert_allclose(
            np.nanmean(list(map(t('cat'), list(map(t('gencov'), self.rg_noint))))), [0, 0], atol=0.02)

    def test_gencov_cat_se(self):
        assert_allclose(np.nanstd(list(map(t('cat'), list(map(t('gencov'), self.rg))))), np.nanmean(
            list(map(t('cat_se'), list(map(t('gencov'), self.rg))))), atol=0.02)

    def test_gencov_cat_se_noint(self):
        assert_allclose(np.nanstd(list(map(t('cat'), list(map(t('gencov'), self.rg_noint))))), np.nanmean(
            list(map(t('cat_se'), list(map(t('gencov'), self.rg_noint))))), atol=0.02)

    def test_gencov_int(self):
        assert_allclose(
            np.nanmean(list(map(t('intercept'), list(map(t('gencov'), self.rg))))), 0, atol=0.1)

    def test_gencov_int_se(self):
        assert_allclose(np.nanmean(list(map(t('intercept_se'), list(map(t('gencov'), self.rg))))), np.nanstd(
            list(map(t('intercept'), list(map(t('gencov'), self.rg))))), atol=0.1)

    def test_hsq_int(self):
        assert_allclose(
            np.nanmean(list(map(t('intercept'), list(map(t('hsq2'), self.rg))))), 1, atol=0.1)

    def test_hsq_int_se(self):
        assert_allclose(np.nanmean(list(map(t('intercept_se'), list(map(t('hsq2'), self.rg))))), np.nanstd(
            list(map(t('intercept'), list(map(t('hsq2'), self.rg))))), atol=0.1)


@attr('h2')
@attr('slow')
class Test_H2_Statistical(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/twold_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.chisq_max = 99999
        h2 = []
        h2_noint = []
        for i in range(N_REP):
            args.intercept_h2 = None
            args.h2 = DIR + '/simulate_test/sumstats/' + str(i)
            args.out = DIR + '/simulate_test/1'
            h2.append(s.estimate_h2(args, log))
            args.intercept_h2 = 1
            h2_noint.append(s.estimate_h2(args, log))

        cls.h2 = h2
        cls.h2_noint = h2_noint

    def test_tot(self):
        assert_allclose(np.nanmean(list(map(t('tot'), self.h2))), 0.9, atol=0.05)

    def test_tot_noint(self):
        assert_allclose(
            np.nanmean(list(map(t('tot'), self.h2_noint))), 0.9, atol=0.05)

    def test_tot_se(self):
        assert_allclose(np.nanmean(list(map(t('tot_se'), self.h2))), np.nanstd(
            list(map(t('tot'), self.h2))), atol=0.05)

    def test_tot_se_noint(self):
        assert_allclose(np.nanmean(list(map(t('tot_se'), self.h2_noint))), np.nanstd(
            list(map(t('tot'), self.h2_noint))), atol=0.05)

    def test_cat(self):
        x = np.nanmean(list(map(t('cat'), self.h2_noint)), axis=0)
        y = np.array((0.3, 0.6)).reshape(x.shape)
        assert_allclose(x, y, atol=0.05)

    def test_cat_noint(self):
        x = np.nanmean(list(map(t('cat'), self.h2_noint)), axis=0)
        y = np.array((0.3, 0.6)).reshape(x.shape)
        assert_allclose(x, y, atol=0.05)

    def test_cat_se(self):
        x = np.nanmean(list(map(t('cat_se'), self.h2)), axis=0)
        y = np.nanstd(list(map(t('cat'), self.h2)), axis=0).reshape(x.shape)
        assert_allclose(x, y, atol=0.05)

    def test_cat_se_noint(self):
        x = np.nanmean(list(map(t('cat_se'), self.h2_noint)), axis=0)
        y = np.nanstd(list(map(t('cat'), self.h2_noint)), axis=0).reshape(x.shape)
        assert_allclose(x, y, atol=0.05)

    def test_coef(self):
        # should be h^2/M = [[0.3, 0.9]] / M
        coef = np.array(((0.3, 0.9))) / self.h2[0].M
        for h in [self.h2, self.h2_noint]:
            assert np.all(np.abs(np.nanmean(list(map(t('coef'), h)), axis=0) - coef) < 1e6)

    def test_coef_se(self):
        for h in [self.h2, self.h2_noint]:
            assert_array_almost_equal(np.nanmean(list(map(t('coef_se'), h)), axis=0),
                                      np.nanstd(list(map(t('coef'), h)), axis=0))

    def test_prop(self):
        for h in [self.h2, self.h2_noint]:
            assert np.all(np.nanmean(list(map(t('prop'), h)), axis=0) - [1/3, 2/3] < 0.02)

    def test_prop_se(self):
        for h in [self.h2, self.h2_noint]:
            assert np.all(np.nanmean(list(map(t('prop_se'), h)), axis=0) - np.nanstd(list(map(t('prop'), h)), axis=0) < 0.02)

    def test_int(self):
        assert_allclose(np.nanmean(list(map(t('intercept'), self.h2))), 1, atol=0.1)

    def test_int_se(self):
        assert_allclose(np.nanstd(list(map(t('intercept'), self.h2))), np.nanmean(
            list(map(t('intercept_se'), self.h2))), atol=0.1)


class Test_Estimate(unittest.TestCase):

    def test_h2_M(self):  # check --M works
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/oneld_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.h2 = DIR + '/simulate_test/sumstats/1'
        args.out = DIR + '/simulate_test/1'
        args.print_cov = True  # right now just check no runtime errors
        args.print_delete_vals = True
        x = s.estimate_h2(args, log)
        args.M = str(
            float(open(DIR + '/simulate_test/ldscore/oneld_onefile.l2.M_5_50').read()))
        y = s.estimate_h2(args, log)
        assert_array_almost_equal(x.tot, y.tot)
        assert_array_almost_equal(x.tot_se, y.tot_se)
        args.M = '1,2'
        assert_raises(ValueError, s.estimate_h2, args, log)
        args.M = 'foo_bar'
        assert_raises(ValueError, s.estimate_h2, args, log)

    def test_h2_ref_ld(self):  # test different ways of reading ref ld
        args = parser.parse_args('')
        args.ref_ld_chr = DIR + '/simulate_test/ldscore/twold_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.h2 = DIR + '/simulate_test/sumstats/555'
        args.out = DIR + '/simulate_test/'
        x = s.estimate_h2(args, log)
        args.ref_ld = DIR + '/simulate_test/ldscore/twold_firstfile,' + \
            DIR + '/simulate_test/ldscore/twold_secondfile'
        y = s.estimate_h2(args, log)
        args.ref_ld_chr = DIR + '/simulate_test/ldscore/twold_firstfile,' + \
            DIR + '/simulate_test/ldscore/twold_secondfile'
        z = s.estimate_h2(args, log)
        assert_almost_equal(x.tot, y.tot)
        assert_array_almost_equal(y.cat, z.cat)
        assert_array_almost_equal(x.prop, y.prop)
        assert_array_almost_equal(y.coef, z.coef)

        assert_array_almost_equal(x.tot_se, y.tot_se)
        assert_array_almost_equal(y.cat_se, z.cat_se)
        assert_array_almost_equal(x.prop_se, y.prop_se)
        assert_array_almost_equal(y.coef_se, z.coef_se)

    # test statistical properties (constrain intercept here)
    def test_rg_M(self):
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/oneld_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.rg = ','.join(
            [DIR + '/simulate_test/sumstats/1' for _ in range(2)])
        args.out = DIR + '/simulate_test/1'
        x = s.estimate_rg(args, log)[0]
        args.M = open(
            DIR + '/simulate_test/ldscore/oneld_onefile.l2.M_5_50', 'r').read().rstrip('\n')
        y = s.estimate_rg(args, log)[0]
        assert_array_almost_equal(x.rg_ratio, y.rg_ratio)
        assert_array_almost_equal(x.rg_se, y.rg_se)
        args.M = '1,2'
        assert_raises(ValueError, s.estimate_rg, args, log)
        args.M = 'foo_bar'
        assert_raises(ValueError, s.estimate_rg, args, log)

    def test_rg_ref_ld(self):
        args = parser.parse_args('')
        args.ref_ld_chr = DIR + '/simulate_test/ldscore/twold_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.rg = ','.join(
            [DIR + '/simulate_test/sumstats/1' for _ in range(2)])
        args.out = DIR + '/simulate_test/1'
        args.print_cov = True  # right now just check no runtime errors
        args.print_delete_vals = True
        x = s.estimate_rg(args, log)[0]
        args.ref_ld = DIR + '/simulate_test/ldscore/twold_firstfile,' + \
            DIR + '/simulate_test/ldscore/twold_secondfile'
        y = s.estimate_rg(args, log)[0]
        args.ref_ld_chr = DIR + '/simulate_test/ldscore/twold_firstfile,' + \
            DIR + '/simulate_test/ldscore/twold_secondfile'
        z = s.estimate_rg(args, log)[0]
        assert_almost_equal(x.rg_ratio, y.rg_ratio)
        assert_almost_equal(y.rg_jknife, z.rg_jknife)
        assert_almost_equal(x.rg_se, y.rg_se)

    def test_no_check_alleles(self):
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/oneld_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.rg = ','.join(
            [DIR + '/simulate_test/sumstats/1' for _ in range(2)])
        args.out = DIR + '/simulate_test/1'
        x = s.estimate_rg(args, log)[0]
        args.no_check_alleles = True
        y = s.estimate_rg(args, log)[0]
        assert_equal(x.rg_ratio, y.rg_ratio)
        assert_almost_equal(x.rg_jknife, y.rg_jknife)
        assert_equal(x.rg_se, y.rg_se)

    def test_twostep_h2(self):
        # make sure two step isn't going crazy
        args = parser.parse_args('')
        args.ref_ld = DIR + '/simulate_test/ldscore/oneld_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.h2 = DIR + '/simulate_test/sumstats/1'
        args.out = DIR + '/simulate_test/1'
        args.chisq_max = 9999999
        args.two_step = 999
        x = s.estimate_h2(args, log)
        args.chisq_max = 9999
        args.two_step = 99999
        y = s.estimate_h2(args, log)
        assert_allclose(x.tot, y.tot, atol=1e-5)

    def test_twostep_rg(self):
        # make sure two step isn't going crazy
        args = parser.parse_args('')
        args.ref_ld_chr = DIR + '/simulate_test/ldscore/oneld_onefile'
        args.w_ld = DIR + '/simulate_test/ldscore/w'
        args.rg = ','.join(
            [DIR + '/simulate_test/sumstats/1' for _ in range(2)])
        args.out = DIR + '/simulate_test/rg'
        args.two_step = 999
        x = s.estimate_rg(args, log)[0]
        args.two_step = 99999
        y = s.estimate_rg(args, log)[0]
        assert_allclose(x.rg_ratio, y.rg_ratio, atol=1e-5)
        assert_allclose(x.gencov.tot, y.gencov.tot, atol=1e-5)
