
import munge_sumstats as munge
import unittest
import numpy as np
import pandas as pd
import nose
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose


class Mock(object):

    '''
    Dumb object for mocking args and log
    '''

    def __init__(self):
        pass

    def log(self, x):
        pass

log = Mock()
args = munge.parser.parse_args('')


class test_p_to_z(unittest.TestCase):

    def setUp(self):
        self.N = pd.Series([1, 2, 3])
        self.P = pd.Series([0.1, 0.1, 0.1])
        self.Z = pd.Series([1.644854, 1.644854, 1.644854])

    def test_p_to_z(self):
        assert_allclose(munge.p_to_z(self.P, self.N), self.Z, atol=1e-5)


class test_check_median(unittest.TestCase):

    def setUp(self):
        self.x = pd.Series([1, 2, 3])

    def test_good_median(self):
        msg = munge.check_median(self.x, 2, 0, 'TEST')
        self.assertEqual(
            msg, 'Median value of TEST was 2.0, which seems sensible.')

    def test_bad_median(self):
        nose.tools.assert_raises(
            ValueError, munge.check_median, self.x, 0, 0.1, 'TEST')


class test_process_n(unittest.TestCase):

    def setUp(self):
        self.dat = pd.DataFrame(['rs1', 'rs2', 'rs3'], columns=['SNP'])
        self.dat_filtered = pd.DataFrame(['rs2', 'rs3'], columns=['SNP'])
        self.dat_filtered['N'] = [1234, 1234.0]
        self.dat_filtered9999 = pd.DataFrame(['rs2', 'rs3'], columns=['SNP'])
        self.dat_filtered9999['N'] = [9999, 9999.0]
        self.args = munge.parser.parse_args('')
        # these flags are either re-set in test cases or should be overridden
        self.args.N = 9999.0
        self.args.N_cas = 9999.0
        self.args.N_con = 9999.0
        self.N_const = pd.Series([1234, 1234, 1234.0])
        self.N = pd.Series([1, 1234, 1234.0])
        self.N_filt = pd.Series([1234, 1234.0])
        self.N9999 = pd.Series([9999.0, 9999, 9999])

    def test_n_col(self):
        self.dat['N'] = self.N
        dat = munge.process_n(self.dat, self.args, log)
        print(dat)
        print(self.dat_filtered)
        assert_frame_equal(dat, self.dat_filtered)

    def test_nstudy(self):
        # should filter on NSTUDY if the --N flag is set, but N gets set to
        # 9999
        self.dat['NSTUDY'] = self.N
        dat = munge.process_n(self.dat, self.args, log)
        assert_frame_equal(dat, self.dat_filtered9999)

    def test_n_cas_con_col(self):
        self.dat['N_CAS'] = self.N
        self.dat['N_CON'] = [0.0, 0, 0]
        dat = munge.process_n(self.dat, self.args, log)
        assert_frame_equal(dat, self.dat_filtered)

    def test_n_flag(self):
        self.args.N = 1234.0
        self.args.N_cas = None
        self.args.N_con = None
        dat = munge.process_n(self.dat, self.args, log)
        assert_series_equal(dat.N, self.N_const, check_names=False)

    def test_n_cas_con_flag(self):
        self.args.N = None
        self.args.N_cas = 1000.0
        self.args.N_con = 234.0
        dat = munge.process_n(self.dat, self.args, log)
        assert_series_equal(dat.N, self.N_const, check_names=False)


def test_filter_pvals():
    P = pd.Series([0, 0.1, 1, 2])
    x = munge.filter_pvals(P, log, args)
    assert_series_equal(x, pd.Series([False, True, True, False]))


def test_single_info():
    dat = pd.Series([0.8, 1, 1])
    x = munge.filter_info(dat, log, args)
    assert_series_equal(x, pd.Series([False, True, True]))


def test_multiple_info():
    i1 = pd.Series([0.8, 1, 1])
    i2 = pd.Series([1.01, 0.5, 9])
    dat = pd.concat([i1, i2], axis=1).reset_index(drop=True)
    dat.columns = ['INFO', 'INFO']
    x = munge.filter_info(dat, log, args)
    assert_series_equal(x, pd.Series([True, False, True]))


def test_filter_frq():
    frq = pd.Series([-1, 0, 0.005, 0.4, 0.6, 0.999, 1, 2])
    x = munge.filter_frq(frq, log, args)
    assert_series_equal(
        x, pd.Series([False, False, False, True, True, False, False, False]))


def test_filter_alleles():
    a = pd.Series(
        ['AC', 'AG', 'CA', 'CT', 'GA', 'GT', 'TC', 'TG', 'DI', 'AAT', 'RA'])
    x = munge.filter_alleles(a)
    y = pd.Series([i < 8 for i in range(11)])
    assert_series_equal(x, y)


class test_allele_merge(unittest.TestCase):

    def setUp(self):
        self.dat = pd.DataFrame(np.transpose([
            ['a', 'b', 'c'],
            ['A', 'T', 'C'],
            ['C', 'G', 'A']]
        ))
        self.dat.columns = ['SNP', 'A1', 'A2']
        self.alleles = pd.DataFrame(np.transpose([
            ['a', 'extra', 'b', 'c'],
            ['AG', 'TC', 'AC', 'AC']]
        ))
        self.alleles.columns = ['SNP', 'MA']

    def test_merge(self):
        x = munge.allele_merge(self.dat, self.alleles, log)
        answer = pd.DataFrame(np.transpose([
            ['a', 'extra', 'b', 'c'],
            ['a', 'a', 'T', 'C'],
            ['a', 'a', 'G', 'A']]
        ))
        answer.columns = ['SNP', 'A1', 'A2']
        answer.loc[[0, 1], ['A1', 'A2']] = float('nan')
        assert_frame_equal(x, answer)


class test_parse_dat(unittest.TestCase):

    def setUp(self):
        dat = pd.DataFrame()
        dat['SNP'] = ['rs' + str(i) for i in range(10)]
        dat['A1'] = ['A' for i in range(10)]
        dat['A2'] = ['G' for i in range(10)]
        dat['INFO'] = np.ones(10)
        dat['FRQ'] = np.ones(10) / 2
        dat['P'] = np.ones(10)
        self.dat = dat
        self.dat_gen = [
            dat.loc[0:4, :], dat.loc[5:9, :].reset_index(drop=True)]
        self.convert_colname = {x: x for x in self.dat_gen[0].columns}
        self.args = munge.parser.parse_args('')

    def tearDown(self):
        args = munge.parser.parse_args('')

    def test_no_alleles(self):
        # test that it doesn't crash with no allele columns and the
        # --no-alleles flag set
        dat = self.dat.drop(['A1', 'A2'], axis=1)
        dat_gen = [dat.loc[0:4, :], dat.loc[5:9, :].reset_index(drop=True)]
        self.args.no_alleles = True
        dat = munge.parse_dat(
            dat_gen, self.convert_colname, None, log, self.args)
        assert_frame_equal(
            dat, self.dat.drop(['INFO', 'FRQ', 'A1', 'A2'], axis=1))

    def test_merge_alleles(self):
        self.args.merge_alleles = True
        merge_alleles = pd.DataFrame()
        merge_alleles['SNP'] = ['rs' + str(i) for i in range(3)]
        merge_alleles['MA'] = ['AG', 'AG', 'AG']
        dat = munge.parse_dat(
            self.dat_gen, self.convert_colname, merge_alleles, log, self.args)
        print(self.dat.loc[0:2, ['SNP', 'A1', 'A2', 'P']])
        assert_frame_equal(dat, self.dat.loc[0:2, ['SNP', 'A1', 'A2', 'P']])

    def test_standard(self):
        dat = munge.parse_dat(
            self.dat_gen, self.convert_colname, None, log, self.args)
        assert_frame_equal(dat, self.dat.drop(['INFO', 'FRQ'], axis=1))

    def test_na(self):
        self.dat.loc[0, 'SNP'] = float('NaN')
        self.dat.loc[1, 'A2'] = float('NaN')
        self.dat_gen = [
            self.dat.loc[0:4, :], self.dat.loc[5:9, :].reset_index(drop=True)]
        dat = munge.parse_dat(
            self.dat_gen, self.convert_colname, None, log, self.args)
        assert_frame_equal(
            dat, self.dat.loc[2:, ['SNP', 'A1', 'A2', 'P']].reset_index(drop=True))


def test_clean_header():
    nose.tools.eq_(munge.clean_header('foo-bar.foo_BaR'), 'FOO_BAR_FOO_BAR')


def test_get_compression_gzip():
    y, x = munge.get_compression('foo.gz')
    nose.tools.eq_(x, 'gzip')
    y, x = munge.get_compression('foo.bz2')
    nose.tools.eq_(x, 'bz2')
    y, x = munge.get_compression('foo.bar')
    nose.tools.eq_(x, None)


class test_parse_flag_cnames(unittest.TestCase):

    def setUp(self):
        self.args = munge.parser.parse_args('')

    def test_basic(self):
        self.args.nstudy = 'nstudy1'
        self.args.snp = 'snp1'
        self.args.N_col = 'n.col1'
        self.args.N_cas_col = 'n-cas.col1'
        self.args.N_con_col = 'n-con.col1'
        self.args.a1 = 'a11'
        self.args.a2 = 'a21'
        self.args.p = 'p1'
        self.args.frq = 'frq1'
        self.args.info = 'info1'
        self.args.info_list = 'info111,info222'
        self.args.signed_sumstats = 'beta1,0'
        x, y = munge.parse_flag_cnames(log, self.args)
        self.assertEqual(y, 0)
        self.assertEqual(x['NSTUDY1'], 'NSTUDY')
        self.assertEqual(x['SNP1'], 'SNP')
        self.assertEqual(x['N_COL1'], 'N')
        self.assertEqual(x['N_CAS_COL1'], 'N_CAS')
        self.assertEqual(x['N_CON_COL1'], 'N_CON')
        self.assertEqual(x['A11'], 'A1')
        self.assertEqual(x['A21'], 'A2')
        self.assertEqual(x['P1'], 'P')
        self.assertEqual(x['FRQ1'], 'FRQ')
        self.assertEqual(x['INFO1'], 'INFO')
        self.assertEqual(x['INFO111'], 'INFO')
        self.assertEqual(x['INFO222'], 'INFO')

    def test_sign_error(self):
        self.args.signed_sumstats = '1,2,3'
        nose.tools.assert_raises(
            ValueError, munge.parse_flag_cnames, log, self.args)
        self.args.signed_sumstats = 'BETA,B'
        nose.tools.assert_raises(
            ValueError, munge.parse_flag_cnames, log, self.args)
        self.args.signed_sumstats = 'BETA'
        nose.tools.assert_raises(
            ValueError, munge.parse_flag_cnames, log, self.args)


class test_cname_map(unittest.TestCase):

    def setUp(self):
        pass

    def test_no_flags(self):
        x = munge.get_cname_map({}, munge.default_cnames, [])
        self.assertEqual(x, munge.default_cnames)

    def test_ignore(self):
        ignore = ['sNp', 'a1']
        flag_cnames = {'SNP': 'SNP', 'ASDF': 'ASDF', 'N': 'FOOBAR'}
        x = munge.get_cname_map(flag_cnames, munge.default_cnames, ignore)
        # check that ignore columns are ignored
        nose.tools.assert_raises(KeyError, x.__getitem__, 'SNP')
        nose.tools.assert_raises(KeyError, x.__getitem__, 'A1')
        # check that flag columns make it into the dict
        self.assertEqual(x['ASDF'], 'ASDF')
        # check that default columns make it into the dict
        self.assertEqual(x['A2'], 'A2')
        # check that flags override default
        self.assertEqual(x['N'], 'FOOBAR')


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        self.args = munge.parser.parse_args('')
        self.args.sumstats = 'test/munge_test/sumstats'
        self.args.out = 'asdf'
        self.args.daner = True

    def test_basic(self):
        x = munge.munge_sumstats(self.args, p=False)
        correct = pd.read_csv(
            'test/munge_test/correct.sumstats', delim_whitespace=True, header=0)
        assert_frame_equal(x, correct)

    def test_merge_alleles(self):
        self.args.merge_alleles = 'test/munge_test/merge_alleles'
        x = munge.munge_sumstats(self.args, p=False)
        correct = pd.read_csv(
            'test/munge_test/correct_merge.sumstats', delim_whitespace=True, header=0)
        assert_frame_equal(x, correct)

    def test_bad_merge_alleles(self):
        self.args.merge_alleles = 'test/munge_test/merge_alleles_bad'
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)

    def test_bad_flags1(self):
        self.args.sumstats = None
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)

    def test_bad_flags2(self):
        self.args.out = None
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)

    def test_bad_flags3(self):
        self.args.merge_alleles = 'foo'
        self.args.no_alleles = 'bar'
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)

    def test_bad_sumstats1(self):
        self.args.signed_sumstats = 'OR,0'
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)

    def test_bad_sumstats1(self):
        self.args.signed_sumstats = 'BETA,0'
        nose.tools.assert_raises(
            ValueError, munge.munge_sumstats, self.args, p=False)
