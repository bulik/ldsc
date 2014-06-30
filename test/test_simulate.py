from __future__ import division
import ldscore.simulate as sim
import unittest
import numpy as np
import nose
from nose_parameterized import parameterized as param


class test_rnorm(unittest.TestCase):
	### TODO add input checking
	@param.expand([([1]),([20]),([30])])
	def test_uvn_shape(self, n):
		'''Test that shape and mean behave correctly'''
		x = sim.rnorm(n, mean=n, var=	0.001)
		assert x.shape == (n,)
		assert np.abs(np.mean(x) - n) < 1

	@param.expand(([1],[-2],[0]))
	def test_uvn_degenerate(self, mean):
		'''If var == 0, all draws should equal the mean'''
		x = sim.rnorm(100, mean=mean, var=0)
		assert np.all(x == np.repeat(mean, 100))
	
	@param.expand(([1],[10],[22]))
	def test_uvn_var(self, var):
		'''Check that the sample variance is close to the theoretical variance'''
		x = sim.rnorm(100000, mean=0, var=var)
		assert np.abs(np.var(x) - var) < 2	
	
	@param.expand(([np.array((1,2,5))],
		[np.array((40,50,2))],
		[np.array((-20,-44,88))],
		))
	def test_mvn_shape(self, mean):
		'''Test that shape and mean behave correctly for MVN'''
		x = sim.rnorm(100, mean=mean, var=np.diag((0.0002,0.0003,0.0004)) )
		assert np.all(x.shape == (100,3) )
		assert np.all(np.abs(mean - np.mean(x,axis=0)) < 1)
	
	@param.expand(([np.diag((2,3))],
		[np.array((1,0.5,0.5,1))],
		[np.array((1,-0.3,-0.3,1))],
		))
	def test_mvn_var(self, var):
		'''Test that MVN gives approx correct variance'''
		var = var.reshape((2,2))
		x = sim.rnorm(100000, mean=np.zeros(2), var=var)
		print np.cov(x.T)
		print np.all(np.cov(x.T) - var < 0.1)


class test_aggregateBeta(unittest.TestCase):
	def test_aggregateBeta(self):
		ldScore = [1,2,3]
		beta = [1,1,1,1,1,1]
		print sim.aggregateBeta(beta, ldScore)
		assert np.all(np.squeeze(sim.aggregateBeta(beta, ldScore)) == [1,2,3])
		
		
class test_pointNormal(unittest.TestCase):
	def test_p_zero(self):
		n = sim.pointNormal(0,size=100, loc=1)
		assert np.all(n == np.ones(100))

	def test_var_and_p(self):
		n = sim.pointNormal(0.5,size=10000, loc=0)
		assert np.abs(np.var(n) - 0.5) < 0.05
		assert np.abs(np.sum(n==0) - 5000) < 1000

class test_sampleSizeMetaAnalysis(unittest.TestCase):
	def test_basic(self):
		for i in xrange(5):
			z1 = np.random.normal(size=5)
			z2 = np.random.normal(size=5)
			n1 = 100
			n2= 1000
			zMeta = sim.sampleSizeMetaAnalysis(n1, n2, z1, z2)
			expMeta = z1*np.sqrt(100/1100) + z2*np.sqrt(1000/1100)
			assert np.all(zMeta - expMeta < 0.01)
  
	@nose.tools.raises(ValueError)
	def test_fixed_variants1(self):
		sim.getFST(0,0.5)
				
	@nose.tools.raises(ValueError)
	def test_fixed_variants2(self):
		sim.getFST(1,0.5)

	@nose.tools.raises(ValueError)
	def test_fixed_variants3(self):
		sim.getFST(0.5,1)

	@nose.tools.raises(ValueError)
	def test_fixed_variants4(self):
		sim.getFST(0.5,0)


class test_getFST(unittest.TestCase):
	def test_basic(self):
		f1 = 0.5
		f2 = 0.3
		print sim.getFST(f1, f2) - 0.08
		assert np.abs(sim.getFST(f1, f2) - 0.08) < 0.0001
	
class test_bivariatePointNormal(unittest.TestCase):
	def test_cor_to_cov(self):
		p1 = 0.5
		p2 = 0.6
		p12 = 0.3
		var1 = 0.9
		var2 = 3
		cov1 =  0.4

	def test_basic(self):
		x = sim.bivariatePointNormal(0.5,0.6,0.3,0.9,3,0.5,size=10)
		assert x.shape == (10,2)

	def test_var(self):
		x = sim.bivariatePointNormal(0.5,0.6,0.3,0.45,1.8,0.2,size=10000)
		assert np.abs(np.var(x[:,0]) - 0.45) < 0.2	
		assert np.abs(np.var(x[:,1]) - 1.8) < 0.2
		assert np.abs(np.corrcoef(x.T)[0,1] - 0.2) < 0.2
		assert np.abs(np.sum(x[:,0]==0) - 5000) < 1000
		assert np.abs(np.sum(x[:,1]==0) - 4000) < 1000
		assert np.abs(np.sum(np.logical_and(x[:,1]!=0,x[:,0]!=0)) - 3000) < 1000
		

	def test_cov_zero(self):
		x = sim.bivariatePointNormal(0.5,0.6,0.3,0.45,1.8,0,size=10000)
		assert np.abs(np.corrcoef(x.T)[0,1]**2 < 0.01)
		assert np.abs(np.var(x[:,0]) - 0.45) < 0.2	
		assert np.abs(np.var(x[:,1]) - 1.8) < 0.2 	