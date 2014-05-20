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
	[np.array([0,-1,-2,0,100]),5],
	[np.array([1,1,1,1,1]),1]
	))
	def test_2d_reg_exact(self, beta, block_size):
		x = np.matrix(np.random.normal(size=500).reshape((100,5)))
		x += 10
		y = np.dot(x, beta).T
		j = jk.LstsqJackknife(x, y, block_size)
		assert np.all(j.est - beta < 10e-6)
		self.assertEqual(j.est.shape, (1,5) )
		assert np.all(j.jknife_est - beta < 10e-6)
		self.assertEqual(j.jknife_est.shape, (1,5) )
		assert np.all(j.jknife_se < 10e-6)
		self.assertEqual(j.jknife_se.shape, (1,5) )
		assert np.all(j.jknife_var < 10e-6)
		self.assertEqual(j.jknife_var.shape, (1,5) )
		assert np.all(j.jknife_cov < 10e-6)
		self.assertEqual(j.jknife_cov.shape, (5,5) )
		
	@param.expand((
		[np.array([0,0,1,0,0]), 10],
		[np.array([11,1.123,-233,0,100]),5],
		[np.array([14,15,1,1,1]),1]
		))
	def test_2d_reg_noise(self, beta, block_size):
		x = np.matrix(np.random.normal(size=500, loc=-10).reshape((100,5)))
		y = np.dot(x, beta).T
		y += np.random.normal(size=100, scale=10e-6).reshape((100,1))
		j = jk.LstsqJackknife(x, y, block_size)
		assert np.all(j.est - beta < 10e-6)
		assert np.all(j.jknife_est - beta < 10e-6)
		assert np.all(j.jknife_se < 10e-3)
		assert np.all(j.jknife_var < 10e-6)
		assert np.all(j.jknife_cov < 10e-6)

	def test_block_action(self):
		x = np.array((-1,0,1,-1,0,1,-1,0,1,))
		y = np.array((-1,0,1, -2,0,2, -3,0,3 ))
		j = jk.LstsqJackknife(x, y, 3)
		xtx_vals = j.block_vals[1]
		xty_vals = j.block_vals[0]
		block_vals = np.array(xty_vals) / np.array(xtx_vals)
		assert np.all(block_vals.reshape(3) - (1,2,3) < 10e-6)
		
	def test_autocor(self):
		x = np.matrix(np.random.normal(size=500, loc=-10).reshape((100,5)))
		y = np.sum(x, axis=1)
		y += np.random.normal(size=100, scale=10e-6).reshape((100,1))
		j = jk.LstsqJackknife(x, y, 20)
		j.autocor(1)		


class Test_RatioJackknife_1D(unittest.TestCase):

	def setUp(self):
		self.numer_delete_vals = np.arange(1,11)
		self.denom_delete_vals = - np.arange(1,11); self.denom_delete_vals[9] += 1
		self.est = -1
		self.num_blocks = len(self.numer_delete_vals)
		self.jknife = jk.RatioJackknife(self.est, self.numer_delete_vals, self.denom_delete_vals)
		
	def test_shapes(self):
		self.assertEqual(self.jknife.est, self.est)
		self.assertEqual(self.jknife.est.shape, (1,))
		self.assertEqual(self.jknife.numer_delete_vals.shape, (self.num_blocks,1))
		assert np.all(self.jknife.numer_delete_vals == np.atleast_2d(self.numer_delete_vals).T)
		self.assertEqual(self.jknife.denom_delete_vals.shape, (self.num_blocks,1))
		assert np.all(self.jknife.denom_delete_vals == np.atleast_2d(self.denom_delete_vals).T)
		self.assertEqual(self.jknife.num_blocks, self.num_blocks)
		self.assertEqual(self.jknife.pseudovalues.shape, self.jknife.numer_delete_vals.shape)

	def test_jknife(self):
		assert np.all(self.jknife.pseudovalues[0:9,:] == -1)
		self.assertEqual(self.jknife.pseudovalues[9,:], 0)
		assert np.abs(self.jknife.jknife_est[0,0] + 0.9) < 10e-6
		assert np.abs(self.jknife.jknife_se - 0.1) < 10e-6
		assert np.abs(self.jknife.jknife_var - 0.01) < 10e-6
		assert np.abs(self.jknife.jknife_cov - 0.01) < 10e-6

				
class Test_RatioJackknife_1D(unittest.TestCase):

	def setUp(self):
		self.numer_delete_vals = np.vstack((np.arange(1,11), 2*np.arange(1,11))).T
		x = - np.arange(1,11); x[9] += 1
		self.denom_delete_vals = np.vstack((x,4*x)).T
		self.est = np.array((-1, -0.5))
		self.num_blocks = self.numer_delete_vals.shape[0]
		self.jknife = jk.RatioJackknife(self.est, self.numer_delete_vals, self.denom_delete_vals)
		
	def test_shapes(self):
		assert np.all(self.jknife.est == self.est)
		self.assertEqual(self.jknife.est.shape, (2,))
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
		assert np.all(np.abs(self.jknife.jknife_est + (0.9, 0.45)) < 10e-6)
		assert np.all(np.abs(self.jknife.jknife_se - (0.1,0.05)) < 10e-6)
		assert np.all(np.abs(self.jknife.jknife_var - (0.01,0.0025)) < 10e-6)
		assert np.all(np.abs(self.jknife.jknife_cov - np.matrix(((0.01,0.005),(0.005,0.0025))) < 10e-6))


class Test_h2_weights(unittest.TestCase):
	
	@param.expand((
		[10,np.array((7,7)),np.ones((10,2)), 0.5],
		[3,np.array((10)),np.ones(10), 1],
		[5,np.array((100,1000)),np.ones((10,2)), 0.01]
		))
	def test_weights(self, M, N, ldScores, hsq):
		x = jk.h2_weights(ldScores, N, M, hsq)
		self.assertTrue(np.all(x == 1 / (1 + hsq*N*ldScores/M)**2))

	
class Test_gencov_weights(unittest.TestCase):

	def test_weights_no_overlap():
		pass
		
	def test_weights_overlap():
		pass

	
class Test_obs_to_liab(unittest.TestCase):
	
	pass
	
	
class test_ldscore_reg(unittest.TestCase):
	
	def setUp(self):
		self.ldScores = np.arange(20)
		self.y = np.arange(20)
		self.block_size = 2
	
	def test_reg(self):
		x = jk.ldscore_reg(self.y, self.ldScores, block_size=self.block_size)		
		self.assertTrue(np.all(x.est - 1 < 10**-6) )
		self.assertTrue(np.all(x.jknife_se - 0 < 10**-6) )
		self.assertTrue(np.all(x.jknife_var - 0 < 10**-6))
		self.assertTrue(np.all(x.jknife_est - 1 < 10**-6))
		
		
class test_ldscore_reg_2D(unittest.TestCase):
	
	def setUp(self):
		self.ldScores = np.vstack((np.arange(6), (0,-2,4,3,-1,-8))).T
		self.y = np.sum(self.ldScores, axis=1)
		self.block_size = 3
	
	def test_reg(self):
		x = jk.ldscore_reg(self.y, self.ldScores, block_size=self.block_size)	
		self.assertTrue(np.all(x.est[0,0:2] - (1,1) < 10**-6) )
		self.assertTrue(np.all(x.jknife_se[0,0:2] - (0,0) < 10**-6) )
		self.assertTrue(np.all(x.jknife_var[0,0:2] - (0,0) < 10**-6))
		self.assertTrue(np.all(x.jknife_est[0,0:2] - (1,1) < 10**-6))
