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
		[np.array([11,1.123,-233,0,100]),20],
		[np.array([14,15,1,1,1]),100]
		))
	def test_2d_reg_noise(self, beta, num_blocks):
		x = np.matrix(np.random.normal(size=500, loc=-10).reshape((100,5)))
		y = np.dot(x, beta).T
		y += np.random.normal(size=100, scale=10e-6).reshape((100,1))
		j = jk.LstsqJackknife(x, y, num_blocks)
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
		pass
		
class Test_infinitesimal_weights(unittest.TestCase):
	
	@param.expand((
		[10,7,np.ones((10,2)), 0.5],
		[3,100,np.ones(10), 1],
		[5,34,np.ones((10,2)), 0.01]
		))
	def test_weights(self, M, N, ldScores, hsq):
		x = jk.infinitesimal_weights(M, ldScores, hsq*N)
		self.assertTrue(np.all(x == 1 / (1 + hsq*N*ldScores/M)**2))
	
	
class test_ldscore_reg(unittest.TestCase):
	
	def setUp(self):
		self.ldScores = np.arange(20)
		self.y = np.arange(20)
		self.num_blocks = 10
	
	def test_reg(self):
		x = jk.ldscore_reg(self.y, self.ldScores, num_blocks=self.num_blocks)		
		self.assertTrue(np.all(x.est - 1 < 10**-6) )
		self.assertTrue(np.all(x.jknife_se - 0 < 10**-6) )
		self.assertTrue(np.all(x.jknife_var - 0 < 10**-6))
		self.assertTrue(np.all(x.jknife_est - 1 < 10**-6))
		
		
class test_ldscore_reg_2D(unittest.TestCase):
	
	def setUp(self):
		self.ldScores = np.vstack((np.arange(6), (0,-2,4,3,-1,-8))).T
		self.y = np.sum(self.ldScores, axis=1)
		self.num_blocks = 2
	
	def test_reg(self):
		x = jk.ldscore_reg(self.y, self.ldScores, num_blocks=self.num_blocks)	
		self.assertTrue(np.all(x.est[0,0:2] - (1,1) < 10**-6) )
		self.assertTrue(np.all(x.jknife_se[0,0:2] - (0,0) < 10**-6) )
		self.assertTrue(np.all(x.jknife_var[0,0:2] - (0,0) < 10**-6))
		self.assertTrue(np.all(x.jknife_est[0,0:2] - (1,1) < 10**-6))
