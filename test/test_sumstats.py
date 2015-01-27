from __future__ import division
import ldscore.sumstats as s
import unittest
import numpy as np
import pandas as pd
import nose
from pandas.util.testing import assert_series_equal, assert_frame_equal
from nose.tools import *
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.plugins.attrib import attr

class Mock(object):
	'''
	Dumb object for mocking args and log
	'''
	def __init__(self):
		pass
	
	def log(self, x):
		pass
		
log = Mock()
args = Mock()

def test_check_condnum():
	x = np.ones((2,2))
	x[1,1] += 1e-5
	args.invert_anyway = False
	assert_raises(ValueError, s._check_ld_condnum, args, log, x)
	args.invert_anyway = True
	s._check_ld_condnum(args, log, x) # no error

def test_check_variance():
	ld = pd.DataFrame(np.vstack((np.ones(3),np.arange(3))).T)
	M_annot = np.array([[1,2]])
	M_annot, ld = s._check_variance(log, M_annot, ld)
	assert_array_equal(M_annot.shape, (1, 1))
	assert_array_equal(M_annot, [[2]])
	assert_series_equal(ld.iloc[:,0], pd.Series([0.0,1,2]))
	
def test_align_alleles():
	beta = pd.Series(np.ones(6))
	alleles = pd.Series(['ACAC','TGTG','GTGT','AGCT','AGTC','TCTC'])
	beta =  s._align_alleles(beta, alleles)
	assert_series_equal(beta, pd.Series([1.0,1,1,-1,1,1]))

def test_filter_bad_alleles():
	alleles = pd.Series(['ATAT','ATAG','DIID','ACAC'])
	bad_alleles = s._filter_alleles(alleles)
	print bad_alleles
	assert_series_equal(bad_alleles, pd.Series([False, False, False, True]))
	
def test_valid_snps():
 x = {'AC', 'AG', 'CA', 'CT', 'GA', 'GT', 'TC', 'TG'}
 assert_equal(x, s.VALID_SNPS)

def test_bases():
	x = set(['A','T','G','C'])
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
	
@attr(speed='slow')
class Test_Simulate_RG(unittest.TestCase):
	
	def setUp(self):
		args = Mock()

	def sim1(self): # test statistical properties (constrain intercept here)
		args.ref_ld = ''
		args.w_ld = ''
		args.rg = ''
		def test(self):
			for i in xrange(1000):
				pass
			
		# unbiasedness
		# correct SE's
	
	def sim2(self): # check that obscure flags work
		args.print_cov = True
		args.print_delete_vals = True 
		args.no_check_alleles	
		args.M = 1234
	
	
			
		