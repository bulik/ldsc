from __future__ import division
import ldscore.regressions as reg
import unittest
import numpy as np
import nose
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises

	
def test_append_intercept():
	x = np.ones((5,2))
	correct_x = np.ones((5,3))
	assert_array_equal(reg.append_intercept(x), correct_x)


def test_kill_brackets():
	x = ' [] [] asdf [] '
	nose.tools.assert_equal(reg.kill_brackets(x), 'asdf')


class Test_h2_obs_to_liab(unittest.TestCase):
	
	def test_bad_data(self):
		assert_raises(ValueError, reg.h2_obs_to_liab, 1 , 1, 0.5)
		assert_raises(ValueError, reg.h2_obs_to_liab, 1 , 0.5, 1)
		assert_raises(ValueError, reg.h2_obs_to_liab, 1 , 0, 0.5)
		assert_raises(ValueError, reg.h2_obs_to_liab, 1 , 0.5, 0)

	def test_approx_scz(self):
		# conversion for a balanced study of a 1% phenotype is about 1/2
		x = reg.h2_obs_to_liab(1, 0.5, 0.01)
		assert_array_almost_equal(x, 0.551907298063)


class Test_gencov_obs_to_liab(unittest.TestCase):
	
	def test_qt(self):
		self.assertEqual(reg.gencov_obs_to_liab(1,None,None,None,None), 1)

	def test_approx_scz(self):
		x = reg.gencov_obs_to_liab(1, 0.5, None, 0.01, None)
		assert_array_almost_equal(x, np.sqrt(0.551907298063))
		x = reg.gencov_obs_to_liab(1, None, 0.5, None, 0.01)
		assert_array_almost_equal(x, np.sqrt(0.551907298063))
		x = reg.gencov_obs_to_liab(1, 0.5, 0.5, 0.01, 0.01)
		assert_array_almost_equal(x, 0.551907298063)


class Test_Hsq_1D(unittest.TestCase):

	def setUp(self):
		self.chisq = np.ones((4,1))
		self.ld = np.ones((4,1))
		self.w_ld = np.ones((4,1))
		self.N = 9*np.ones((4,1))
		self.M = np.matrix((7))
		self.hsq = reg.Hsq(self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=2, intercept=1)

	def test_weights(self):
		hsq = 0.5
		w = reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, hsq)
		assert_array_equal(w.shape, self.ld.shape)
		assert_array_almost_equal(w[0,0], 1 / np.square(1+ hsq*self.N/self.M))
		# test that it deals correctly with out-of-bounds h2
		assert_array_almost_equal(reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 1),\
			reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 2))
		assert_array_almost_equal(reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, 0),\
			reg.Hsq.weights(self.ld, self.w_ld, self.N, self.M, -1))
		
	def test_summarize_chisq(self):
		chisq = np.arange(100).reshape((100,1))
		mean_chisq, lambda_gc = self.hsq._summarize_chisq(chisq)
		assert_array_almost_equal(mean_chisq, 49.5)
		assert_array_almost_equal(lambda_gc, 108.81512420312156)
		
	def test_coef(self):
		pass
	
	def test_cat_hsq(self):
		pass
		
	def test_hsq(self):
		pass
	
	def test_prop_hsq(self):
		pass
		
	def test_enrichment(self):
		pass
		
	def test_summary(self):
		# not much to test; we can at least make sure no errors at runtime
		self.hsq.summary(['asdf'])
		self.ld += np.arange(4).reshape((4,1))
		self.chisq += np.arange(4).reshape((4,1))	
		hsq = reg.Hsq(self.chisq, self.ld, self.w_ld, self.N, self.M, n_blocks=2)
		self.hsq.summary(['asdf'])
		
'''
				

class Test_Hsq_1D(unittest.TestCase):

	def setUp(self):
		self.ldScores = np.matrix(np.arange(20)).reshape((20,1))
		self.w = np.matrix(np.ones(20)).reshape((20,1))
		self.y = np.matrix(np.arange(20)).reshape((20,1))
		self.N = np.matrix(np.ones(20)).reshape((20,1))
		self.M = np.matrix(1)
		self.num_blocks = 10

	def test_reg(self):
		x = jk.Hsq(self.y, self.ldScores, self.w, self.N, self.M, num_blocks=self.num_blocks)		
		self.assertTrue(np.all( np.abs(x.tot_hsq - 1) < 1e-6) )
		self.assertTrue(np.all( np.abs(x.tot_hsq_se - 0) < 1e-6) )
		self.assertTrue(np.all( np.abs(x._jknife.jknife_var - 0) < 1e-6))


class Test_Hsq_2D(unittest.TestCase):

	def setUp(self):
		self.ldScores = np.matrix(np.vstack((np.arange(6), (0,-2,4,3,-1,-8))).T).reshape((6,2))
		self.y = np.sum(self.ldScores, axis=1)
		self.w = np.matrix(np.ones(6)).reshape((6,1))
		self.num_blocks = 2
		self.N = np.matrix(np.ones(6)).reshape((6,1))
		self.M = np.matrix((1,1)).reshape((1,2))
	
	def test_reg(self):
		x = jk.Hsq(self.y, self.ldScores, self.w, self.N, self.M, num_blocks=self.num_blocks)	
		self.assertTrue(np.all( np.abs(x.cat_hsq[0,0:2] - (1,1)) < 1e-6) )
		self.assertTrue(np.all( np.abs(x.cat_hsq_se[0,0:2] - (0,0)) < 1e-6) )
		self.assertTrue(np.all( np.abs(x._jknife.jknife_var[0,0:2] - (0,0)) < 1e-6))
		self.assertTrue(np.all( np.abs(x._jknife.jknife_est[0,0:2] - (1,1)) < 1e-6))
'''