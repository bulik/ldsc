from __future__ import division
import ldscore.regressions as reg
import unittest
import numpy as np
import nose
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises
from nose_parameterized import parameterized as param



class Test_append_intercept(unittest.TestCase):
	
	def test_simple(self):
		x = np.matrix(np.ones(10).reshape((5,2)))
		correct_x = np.matrix(np.ones(15).reshape((5,3)))
		assert np.all(reg._append_intercept(x) == correct_x)

'''


class Test_weight(unittest.TestCase):
				
	def test_symmetric(self):
		x = np.matrix(np.ones((2,2)))
		w = np.matrix((0.25,0.75)).T
		correct_x = np.matrix(( (0.5, 0.5), (0.8660254, 0.8660254)) )
		print jk._weight(x, w)
		assert np.all(np.abs(jk._weight(x, w) - correct_x) < 1e-6)

	def test_asymmetric(self):
		x = np.matrix(np.ones((10,2)))
		w = np.matrix(np.arange(10)+1 ).T
		wp = np.sqrt(w / float(np.sum(w)))
		correct_x = np.hstack((wp,wp))
		print jk._weight(x, w)
		print correct_x
		assert np.all(np.abs(jk._weight(x, w) - correct_x) < 1e-6)

	@param.expand((
		[10,22,np.arange(10)+1, 0.5],
		[3,33,np.ones(10)+1, 1],
		))
	def test_weights(self, N, M, ld, hsq):
		x = jk._hsq_weights(ld, np.ones_like(ld), N, M, hsq)		
		assert np.all( np.abs(x - 1 / (1 + hsq*N*ld/M)**2) < 1e-6)


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