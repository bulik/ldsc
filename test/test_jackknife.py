from __future__ import division
import ldsc.jackknife as jk
import unittest
import numpy as np
import nose
from nose_parameterized import parameterized as param


class TestLstsqJackknife(unittest.TestCase):
	
	def setUp(self):	
		pass

	@param.expand((
	[np.array([1,2,3,4,5]), 10],
	[np.array([0,-1,-2,0,100]),20],
	[np.array([1,1,1,1,1]),100]
	))
	def test_2d_reg_exact(self, beta, num_blocks):
		x = np.matrix(np.random.normal(size=500).reshape((100,5)))
		x += 10
		y = np.dot(x, beta).T
		j = jk.LstsqJackknife(x, y, num_blocks)
		assert np.all(np.abs(j.est - beta) < 10e-6)
		self.assertEqual(j.est.shape, (1,5) )
		assert np.all(np.abs(j.jknife_est - beta) < 1e-6)
		self.assertEqual(j.jknife_est.shape, (1,5) )
		assert np.all(np.abs(j.jknife_se) < 1e-6)
		self.assertEqual(j.jknife_se.shape, (1,5) )
		assert np.all(np.abs(j.jknife_var) < 1e-6)
		self.assertEqual(j.jknife_var.shape, (1,5) )
		assert np.all(np.abs(j.jknife_cov) < 1e-6)
		self.assertEqual(j.jknife_cov.shape, (5,5) )
		
	@param.expand((
		[np.array([0,0,1,0,0]), 10],
		[np.array([11,1.123,-233,0,100]),20],
		[np.array([14,15,1,1,1]),100]
		))
	def test_2d_reg_noise(self, beta, num_blocks):
		x = np.matrix(np.random.normal(size=500, loc=-10).reshape((100,5)))
		y = np.dot(x, beta).T
		y += np.random.normal(size=100, scale=1e-6).reshape((100,1))
		j = jk.LstsqJackknife(x, y, num_blocks)
		assert np.all(np.abs(j.est - beta) < 1e-6)
		assert np.all(np.abs(j.jknife_est - beta) < 1e-6)
		assert np.all(np.abs(j.jknife_se) < 1e-3)
		assert np.all(np.abs(j.jknife_var) < 1e-6)
		assert np.all(np.abs(j.jknife_cov) < 1e-6)

	def test_block_action(self):
		x = np.array((-1,0,1,-1,0,1,-1,0,1,))
		y = np.array((-1,0,1, -2,0,2, -3,0,3 ))
		j = jk.LstsqJackknife(x, y, 3)
		xtx_vals = j.block_vals[1]
		xty_vals = j.block_vals[0]
		block_vals = np.array(xty_vals) / np.array(xtx_vals)
		assert np.all( np.abs(block_vals.reshape(3) - (1,2,3)) < 1e-6)
		
	def test_autocor(self):
		x = np.matrix(np.random.normal(size=500, loc=-10).reshape((100,5)))
		y = np.sum(x, axis=1)
		y += np.random.normal(size=100, scale=1e-6).reshape((100,1))
		j = jk.LstsqJackknife(x, y, 20)
		j.autocor(1)		


class Test_RatioJackknife_1D(unittest.TestCase):

	def setUp(self):
		self.numer_delete_vals = np.matrix(np.arange(1,11)).T
		self.denom_delete_vals = - np.matrix(np.arange(1,11)).T; self.denom_delete_vals[9,0] += 1
		self.est = np.matrix(-1)
		self.num_blocks = self.numer_delete_vals.shape[0]
		self.jknife = jk.RatioJackknife(self.est, self.numer_delete_vals, self.denom_delete_vals)
		
	def test_shapes(self):
		self.assertEqual(self.jknife.est, self.est)
		self.assertEqual(self.jknife.est.shape, (1,1))
		self.assertEqual(self.jknife.numer_delete_vals.shape, (self.num_blocks,1))
		assert np.all(self.jknife.numer_delete_vals == self.numer_delete_vals)
		self.assertEqual(self.jknife.denom_delete_vals.shape, (self.num_blocks,1))
		assert np.all(self.jknife.denom_delete_vals == self.denom_delete_vals)
		self.assertEqual(self.jknife.num_blocks, self.num_blocks)
		self.assertEqual(self.jknife.pseudovalues.shape, self.jknife.numer_delete_vals.shape)

	def test_jknife(self):
		assert np.all(self.jknife.pseudovalues[0:9,:] == -1)
		self.assertEqual(self.jknife.pseudovalues[9,:], 0)
		assert np.abs(self.jknife.jknife_est[0,0] + 0.9) < 1e-6
		assert np.abs(self.jknife.jknife_se - 0.1) < 1e-6
		assert np.abs(self.jknife.jknife_var - 0.01) < 1e-6
		assert np.abs(self.jknife.jknife_cov - 0.01) < 1e-6

				
class Test_RatioJackknife_2D(unittest.TestCase):

	def setUp(self):
		self.numer_delete_vals = np.matrix(np.vstack((np.arange(1,11), 2*np.arange(1,11)))).T
		x = - np.arange(1,11); x[9] += 1
		self.denom_delete_vals = np.vstack((x,4*x)).T
		self.est = np.matrix((-1, -0.5))
		self.num_blocks = self.numer_delete_vals.shape[0]
		self.jknife = jk.RatioJackknife(self.est, self.numer_delete_vals, self.denom_delete_vals)
		
	def test_shapes(self):
		assert np.all(self.jknife.est == self.est)
		self.assertEqual(self.jknife.est.shape, (1,2))
		self.assertEqual(self.jknife.numer_delete_vals.shape, (self.num_blocks,2))
		assert np.all(self.jknife.numer_delete_vals == self.numer_delete_vals)
		self.assertEqual(self.jknife.denom_delete_vals.shape, (self.num_blocks,2))
		print self.jknife.denom_delete_vals
		assert np.all(self.jknife.denom_delete_vals == self.denom_delete_vals)
		self.assertEqual(self.jknife.num_blocks, self.num_blocks)
		self.assertEqual(self.jknife.pseudovalues.shape, self.jknife.numer_delete_vals.shape)
	
	def test_jknife(self):
		assert np.all(self.jknife.pseudovalues[0:9,0] == -1)
		self.assertEqual(self.jknife.pseudovalues[9,0], 0)
		assert np.all(self.jknife.pseudovalues[0:9,1] == -0.5)
		self.assertEqual(self.jknife.pseudovalues[9,1], 0)
		assert np.all(np.abs(self.jknife.jknife_est + (0.9, 0.45)) < 1e-6)
		assert np.all(np.abs(self.jknife.jknife_se - (0.1,0.05)) < 1e-6)
		assert np.all(np.abs(self.jknife.jknife_var - (0.01,0.0025)) < 1e-6)
		assert np.all(np.abs(self.jknife.jknife_cov - np.matrix(((0.01,0.005),(0.005,0.0025)))) < 1e-6)


class Test_append_intercept(unittest.TestCase):
	
	def test_simple(self):
		x = np.matrix(np.ones(10).reshape((5,2)))
		correct_x = np.matrix(np.ones(15).reshape((5,3)))
		assert np.all(jk._append_intercept(x) == correct_x)


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
		[5,100,np.arange(100)+1, 0.01]
		))
	def test_weights(self, N, M, ld, hsq):
		x = jk._hsq_weights(ld, np.ones_like(ld), N, M, hsq)		
		assert np.all(x - 1 / (1 + hsq*N*ld/M)**2 < 1e-6)


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
class Test_gencov_weights(unittest.TestCase):


def test_basic(self):
		pass
			


class Test_obs_to_liab(unittest.TestCase):
	
	pass
	
'''
