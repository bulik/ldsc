from __future__ import division
import ldsc.gwas as gw
import unittest
import numpy as np
import scipy.stats as spstats
import nose
from nose_parameterized import parameterized as param

'''
TODO : currently no checks that insert() is working properly 
possibly solution: add some attribute that indicates which indivs were inserted

'''

BIG_N = 15000

class test_QT_GWAS_basic(unittest.TestCase):
	'''Test basic functions, e.g., that genotypes are correctly normalized, etc'''
	def setUp(self):		
		self.M = 3
		self.mafs = [0.5, 0.5, 0.5]
		self.betas = (1, -1, 0)
		self.N = BIG_N
		self.hsqs = 0.5
		self.gwas = gw.QT_GWAS(self.M, self.N, self.mafs, self.betas, self.hsqs)

	def test_init(self):
		self.assertEqual(self.gwas.M, self.M)
		self.assertEqual(self.gwas.N, self.N)
		self.assertTrue( np.all(self.gwas.mafs == self.mafs) )
		self.assertEqual(self.hsqs, self.gwas.hsqs)
		self.assertEqual(1 - self.hsqs, self.gwas.e_var)
		self.assertTrue(np.abs(np.dot(self.gwas.betas.T, self.gwas.betas) - self.hsqs < 10**-6))
	
	def test_bv(self):
		fake_geno = np.ones((self.N,self.M))
		self.assertTrue(np.all(self.gwas.__bv__(fake_geno) == np.dot(fake_geno, self.gwas.betas)) )
		
	@param.expand(([1,2,3],
	[-4,5,1],
	[11,12,23]))
	def test_calc_pheno(self, bv, ev, correct_pheno):
		self.assertEqual(self.gwas.__calc_pheno__(bv, ev), correct_pheno)
	
 	def test_geno(self):
 		self.gwas.sample()	
 		geno = self.gwas.geno
 		self.assertEqual(geno.shape, (self.N, self.M))
		self.assertTrue(np.all(abs(geno.mean(axis=0)) < 0.1))
 		self.assertTrue(np.all(abs(geno.var(axis=0)-1) < 0.05))
	
	def test_hsq_0(self):
		self.gwas = gw.QT_GWAS(self.M, self.N, self.mafs, self.betas, 0)
		self.gwas.sample()
		self.assertTrue(np.all(self.gwas.bv == np.zeros(self.M)))
		self.assertTrue(np.abs(np.var(self.gwas.pheno) - 1) < 0.2)
		self.assertTrue(np.abs(np.var(self.gwas.ev) - 1) < 0.2)
		self.assertTrue(np.all(self.gwas.pheno == self.gwas.ev))
	
	def test_hsq_1(self):
		self.gwas = gw.QT_GWAS(self.M, self.N, self.mafs, self.betas, 1)
		self.gwas.sample()
		self.assertTrue(np.all(self.gwas.ev == np.zeros(self.M)))
		self.assertTrue(np.abs(np.var(self.gwas.pheno) - 1) < 0.2)
		self.assertTrue(np.abs(np.var(self.gwas.bv) - 1) < 0.2)
		self.assertTrue(np.all(self.gwas.pheno == self.gwas.bv))

	def test_sumstats(self):
		self.gwas.sample()
		betahat = self.gwas.betahat
		chisq = self.gwas.chisq
		self.assertEqual(chisq.shape, (3,1))
		self.assertEqual(betahat.shape, (3,1))
		self.assertTrue(chisq[0] > 1000)
		self.assertTrue(chisq[1] > 1000)
		self.assertTrue(chisq[2] < 30)


class test_QT_GWAS_vectorized(unittest.TestCase):
	'''Test with multiple phenotypes in a single GWAS'''
	def setUp(self):
		self.M = 3; 
		self.N = BIG_N
		self.mafs=[0.5,0.5,0.5]
		self.betas = np.matrix((1,0, -1,1, 0,-1)).reshape((self.M, 2))
		self.hsqs = np.matrix((0.9,0.5))
		self.e_var = np.matrix((0.1,0.03,0.03,0.5)).reshape((2,2))
		self.gwas = gw.QT_GWAS(self.M, self.N, self.mafs, self.betas, self.hsqs, self.e_var)

	def test_init(self):
		self.assertEqual(self.betas.shape, (self.M, 2))
		bb = np.diag(np.dot(self.gwas.betas.T, self.gwas.betas))
		self.assertTrue(np.all(np.abs(bb - self.hsqs < 10**-6)))

	def test_shapes(self):
		self.gwas.sample()
		self.assertEqual(self.gwas.pheno.shape, (self.N, 2))
		self.assertEqual(self.gwas.bv.shape, (self.N, 2))
		self.assertEqual(self.gwas.ev.shape, (self.N, 2))
		self.assertEqual(self.gwas.geno.shape, (self.N, self.M))
		self.assertEqual(self.gwas.betas.shape, (self.M, 2))
		self.assertEqual(self.gwas.hsqs.shape, (1,2))
		self.assertEqual(self.gwas.e_var.shape, (2,2))
		
	def test_vars(self):
		self.gwas.sample()
		p1_cov = np.var(self.gwas.pheno[:,0])
		p2_cov = np.var(self.gwas.pheno[:,1])
		self.assertTrue(np.abs(p1_cov - 1) < 0.2)
		self.assertTrue(np.abs(p2_cov - 1) < 0.2)
		ev_cov = np.cov(self.gwas.ev.T)
		self.assertTrue(np.all(np.abs(ev_cov - self.e_var) <  0.1))
		
	def test_sumstats(self):
		chisq = self.gwas.chisq
		betahat = self.gwas.betahat
		self.assertEqual(chisq.shape, (self.M,1))
		self.assertEqual(betahat.shape, (self.M,1))
		self.assertTrue(chisq[0] > 1000)
		self.assertTrue(chisq[1] > 1000)
		self.assertTrue(chisq[2] < 30)

		
class test_QT_insert(unittest.TestCase):
 	
 	def setUp(self):
 		mafs = np.ones(10)/2
 		betas1 = np.matrix(np.random.normal(size=20)).reshape((10,2))
 		betas2 = betas1[:,1]
 		hsqs1 = np.array((0.5, 0.9))
 		hsqs2 = 0.9
 		e_var1 = np.matrix((0.5,0.03,0.03,0.1)).reshape((2,2))
 		self.gwas1 = gw.QT_GWAS(10, 5, mafs, betas1, hsqs1, e_var1)
		self.gwas1.sample()
 		self.gwas2 = gw.QT_GWAS(10, 10, mafs, betas2, hsqs2)
		self.extract = self.gwas1.__extract__(np.arange(5), 1)
 		self.gwas2.__insert__(self.extract) ### TODO this interface sucks
 		self.gwas2.sample()
 	
 	def test_ev(self):
 		self.assertTrue(np.all(self.gwas2.ev[0:5,0] == self.gwas1.ev[:,1]))
 		
 	def test_bv(self):
 		self.assertTrue(np.all(self.gwas2.bv[0:5,0] == self.gwas1.bv[:,1]))
 	
 	def test_geno(self):
 		self.assertTrue(np.all(self.gwas2.geno[0:5,:] == self.gwas1.geno))
 	
 	
class test_CC_GWAS(unittest.TestCase):

	def setUp(self):
		self.M = 3
		self.mafs = [0.5, 0.5, 0.5]
		self.betas = (1, -1, 0)
		self.N_cas = 10; self.N_con = 10
		self.P = 0.5
		self.hsqs = 0.5
		self.chunk_size = 10
		self.gwas = gw.CC_GWAS(self.M, self.N_cas, self.N_con, self.mafs, self.betas, 
			self.hsqs, self.P, chunk_size=self.chunk_size)
		self.gwas.sample()
		
	@param.expand(([0.01, 2.3263478740408408, 0.02665214220345808],
	[0.1, 1.2815515655446004, 0.17549833193248685],
	[0.5, 0.0, 0.3989422804014327]))
	def test_z_thresh(self, P, correct_thresh, correct_z):
		gwas = gw.CC_GWAS(self.M, self.N_cas, self.N_con, self.mafs, self.betas, self.hsqs, 
			P, chunk_size=self.chunk_size)
		self.assertEqual(gwas.thresh, correct_thresh)
		self.assertEqual(gwas.__z__, correct_z)
		
	def test_init(self):
		self.assertEqual(self.gwas.P, self.P)
		self.assertEqual(self.gwas.N_cas, self.N_cas)
		self.assertEqual(self.gwas.N_con, self.N_con)
		self.assertEqual(self.gwas.N, self.N_cas+self.N_con)
		self.assertEqual(self.gwas.K, self.N_cas/(self.N_cas + self.N_con))
		self.assertEqual(self.gwas.chunk_size, self.chunk_size)

	@param.expand(([0.5,0.5,1], [-0.4,3,1], [-2,-2,0], [0,0.1,1]))
	def test_calc_pheno(self, bv, ev, correct_pheno):	
		self.assertEqual(self.gwas.__calc_pheno__(bv, ev), correct_pheno)

	@param.expand(( [ (0.25,0.25), (0.25,0.25), (1,0) ],
	[ (-0.25,1), (-0.25,1), (0,1) ],
	[ (5,4), (3,2), (1,1) ],
	[ (-0.25,-1), (-0.25,-1), (0,0) ]
	))
	def test_calc_pheno_vectorized(self, bv, ev, correct_pheno):	
		bv = np.matrix(bv).reshape((1,2))
		ev = np.matrix(ev).reshape((1,2))
		correct_pheno = np.matrix(correct_pheno).reshape((1,2))
		self.gwas.thresh = np.matrix(spstats.norm.isf((0.5,0.1))).reshape((1,2))		
		self.assertTrue(np.all(self.gwas.__calc_pheno__(bv, ev) == correct_pheno))

	@param.expand(([np.matrix((1,0,1)).reshape((3,1)), np.array((True,False,True))], 
	[np.matrix((0,0,0)).reshape((3,1)), np.array((False,False,False))],
	[np.matrix((1,1,1)).reshape((3,1)), np.array((True,True,True))],
	[np.matrix((1,0,1,0,1,0)).reshape((3,2)), np.array((True,True,True))],
	[np.matrix((0,0,0,1,0,1)).reshape((3,2)), np.array((False,False,False))]
	))
	def test_ascertain_cases(self, phenos, correct_phenos):
		draw = (0,0,0,phenos)
		self.assertTrue(np.all(self.gwas.__ascertain_cases__(draw) == correct_phenos))
	
	@param.expand(([np.matrix((0,0,1)).reshape((3,1))], 
	[np.matrix((0,0,0)).reshape((3,1))],
	[np.matrix((1,1,1)).reshape((3,1))],
	[np.matrix((0,0,0,0,1,0)).reshape((3,2))],
	[np.matrix((0,0,0,1,0,1)).reshape((3,2))],
	[np.matrix((1,0,0,1,0,1)).reshape((3,2))]
	))
	def test_ascertain_controls(self, phenos):
		draw = (0,0,0,phenos)
		correct_phenos = np.squeeze(np.asarray(np.logical_not(np.prod(phenos,axis=1))))
		self.assertTrue(np.all(self.gwas.__ascertain_controls__(draw) == correct_phenos))


class test_QT_QT_GWAS(unittest.TestCase):
	
	def setUp(self):
		self.M = 3
		self.mafs = [0.5, 0.5, 0.5]
		self.beta1 = (1, -1, 0)
		self.beta2 = (0,-1,1)
		self.N1 = BIG_N
		self.N2 = BIG_N + 1
		self.hsq1 = 0.9
		self.hsq2 = 0.5
		self.e_var = np.matrix((0.1,0.03,0.03,0.5)).reshape((2,2))
		
	@param.expand(([0],[5],[10]))
	def test_init(self, overlap):
		self.gwas = gw.QT_QT_GWAS(self.M, self.N1, self.N2, self.mafs, self.beta1, self.beta2,
			self.hsq1, self.hsq2, self.e_var, overlap)
		self.gwas.sample()
		self.assertEqual(self.N1, self.gwas.gwas1.N)
		self.assertEqual(self.N2, self.gwas.gwas2.N)
		self.assertEqual(self.M, self.gwas.gwas1.M)
		self.assertEqual(self.M, self.gwas.gwas2.M)
		self.assertTrue(np.all(self.mafs == self.gwas.gwas1.mafs))
		self.assertTrue(np.all(self.mafs == self.gwas.gwas2.mafs))
		self.assertTrue(np.all(np.matrix((self.hsq1, self.hsq2)) == self.gwas.gwas1.hsqs))
		self.assertEqual(self.hsq2, self.gwas.gwas2.hsqs)
		self.assertEqual(overlap, self.gwas.overlap)
		self.assertTrue(np.all(self.e_var == self.gwas.gwas1.e_var))
		self.assertEqual(self.e_var[1,1], self.gwas.gwas2.e_var)

	@param.expand(([0],[5],[10]))
	def test_sample(self, overlap):
		self.gwas = gw.QT_QT_GWAS(self.M, self.N1, self.N2, self.mafs, self.beta1, self.beta2,
			self.hsq1, self.hsq2, self.e_var, overlap)
		self.gwas.sample()
		self.assertEqual(self.gwas.gwas1.__current_N__, self.gwas.gwas1.N)
		self.assertEqual(self.gwas.gwas2.__current_N__, self.gwas.gwas2.N)

	@param.expand(([0],[5],[10]))
	def test_sumstats(self, overlap):
		self.gwas = gw.QT_QT_GWAS(self.M, self.N1, self.N2, self.mafs, self.beta1, self.beta2,
			self.hsq1, self.hsq2, self.e_var, overlap)
		chisq = self.gwas.chisq
		betahat = self.gwas.betahat
		self.assertEqual(chisq.shape, (self.M,2))
		self.assertEqual(betahat.shape, (self.M,2))
		self.assertTrue(chisq[0,0] > 1000)
		self.assertTrue(chisq[1,0] > 1000)
		self.assertTrue(chisq[2,0] < 30)		
		self.assertTrue(chisq[0,1] < 30)
		self.assertTrue(chisq[1,1] > 500)
		self.assertTrue(chisq[2,1] > 500)		
		
		
class test_QT_CC_GWAS(unittest.TestCase):
	
	def setUp(self):
		self.M = 3
		self.mafs = [0.5, 0.5, 0.5]
		self.beta1 = (1, -1, 0)
		self.beta2 = (0,-1,1)
		self.N1 = BIG_N
		self.N_cas = 1000
		self.N_con = 1000
		self.hsq1 = 0.8
		self.hsq2 = 0.9
		self.P = 0.5
		self.e_var = np.matrix((0.2,0.03,0.03,0.1)).reshape((2,2))
		self.overlap = 10
		self.chunk_size = 100

	@param.expand(([0],[1000],[2000]))
	def test_init(self, overlap):
		self.gwas = gw.QT_CC_GWAS(self.M, self.N1, self.N_cas, self.N_con, self.mafs, 
			self.beta1, self.beta2, self.hsq1, self.hsq2, self.P, self.e_var, overlap, 
			self.chunk_size)
		self.gwas.sample()
		self.assertEqual(self.N1, self.gwas.gwas1.N)
		self.assertEqual(self.N_cas + self.N_con, self.gwas.gwas2.N)
		self.assertEqual(self.M, self.gwas.gwas1.M)
		self.assertEqual(self.M, self.gwas.gwas2.M)
		self.assertTrue(np.all(self.mafs == self.gwas.gwas1.mafs))
		self.assertTrue(np.all(self.mafs == self.gwas.gwas2.mafs))
		self.assertTrue(np.all(np.matrix((self.hsq1, self.hsq2)) == self.gwas.gwas1.hsqs))
		self.assertEqual(self.hsq2, self.gwas.gwas2.hsqs)
		self.assertEqual(overlap, self.gwas.overlap)
		self.assertTrue(np.all(self.e_var == self.gwas.gwas1.e_var))
		self.assertTrue(self.e_var[1,1] - self.gwas.gwas2.e_var < 10**-6)

	@param.expand(([0],[1000],[2000]))
	def test_sample(self, overlap):
		self.gwas = gw.QT_CC_GWAS(self.M, self.N1, self.N_cas, self.N_con, self.mafs, 
			self.beta1, self.beta2, self.hsq1, self.hsq2, self.P, self.e_var, overlap, 
			self.chunk_size)
		self.gwas.sample()
		self.assertEqual(self.gwas.gwas2.__current_N_cas__, self.N_cas)
		self.assertEqual(self.gwas.gwas2.__current_N_con__, self.N_con)
		self.assertEqual(self.gwas.gwas2.__current_N__, self.N_cas + self.N_con)
	
	@param.expand(([0],[1000],[2000]))
	def test_sumstats(self, overlap):
		self.gwas = gw.QT_CC_GWAS(self.M, self.N1, self.N_cas, self.N_con, self.mafs, 
			self.beta1, self.beta2, self.hsq1, self.hsq2, self.P, self.e_var, overlap, 
			self.chunk_size)
		self.gwas.sample()
		chisq = self.gwas.chisq
		betahat = self.gwas.betahat
		self.assertEqual(chisq.shape, (self.M,2))
		self.assertEqual(betahat.shape, (self.M,2))
		self.assertTrue(chisq[0,0] > 1000)
		self.assertTrue(chisq[1,0] > 1000)
		self.assertTrue(chisq[2,0] < 30)		
		self.assertTrue(chisq[0,1] < 30)
		self.assertTrue(chisq[1,1] > 50)
		self.assertTrue(chisq[2,1] > 50)	

'''
def __init__(self,M,N_cas,N_con,mafs,betas,hsq,e_var,P,overlap,chunk_size=4000):

'''

class test_CC_CC_GWAS(unittest.TestCase):

	def setUp(self):
		self.M = 3
		self.N_cas = np.array((100, 101))
		self.N_con = np.array((102, 103))
		self.mafs = [0.5, 0.5, 0.5]
		self.betas = np.matrix((1,0,-1,1,0,1)).reshape((3,2))
		self.hsqs = np.array((0.99,0.95))
		self.P = np.array((0.3,0.2))
		self.chunk_size = 40
		self.e_var = np.matrix((0.01,0,0,0.05)).reshape((2,2))
		
	@param.expand(([0],[50],[101]))
	def test_init(self, overlap):
		self.gwas = gw.CC_CC_GWAS(self.M, self.N_cas, self.N_con, self.mafs, self.betas, 
			self.hsqs,self.P, self.e_var,  overlap, self.chunk_size)
		self.gwas.sample()
		self.assertEqual(self.N_cas[0], self.gwas.gwas1.N_cas)		
		self.assertEqual(self.N_cas[1], self.gwas.gwas2.N_cas)
		self.assertEqual(self.N_con[0], self.gwas.gwas1.N_con)		
		self.assertEqual(self.N_con[1], self.gwas.gwas2.N_con)
		self.assertEqual(self.M, self.gwas.gwas1.M)
		self.assertEqual(self.M, self.gwas.gwas2.M)
		self.assertTrue(np.all(self.mafs == self.gwas.gwas1.mafs))
		self.assertTrue(np.all(self.mafs == self.gwas.gwas2.mafs))
		self.assertEqual(overlap, self.gwas.overlap)
		self.assertTrue(np.all(self.gwas.gwas1.e_var == self.e_var))
		self.assertTrue(np.all(self.gwas.gwas2.e_var == self.e_var))

	@param.expand(([0],[50],[101]))
	def test_sample(self, overlap):
		self.gwas = gw.CC_CC_GWAS(self.M, self.N_cas, self.N_con, self.mafs, self.betas, 
			self.hsqs,self.P, self.e_var,  overlap, self.chunk_size)
		self.gwas.sample()
		self.assertEqual(self.gwas.gwas1.__current_N_cas__, self.N_cas[0])
		self.assertEqual(self.gwas.gwas1.__current_N_con__, self.N_con[0])
		self.assertEqual(self.gwas.gwas1.__current_N__, self.N_cas[0] + self.N_con[0])
		self.assertEqual(self.gwas.gwas2.__current_N_cas__, self.N_cas[1])
		self.assertEqual(self.gwas.gwas2.__current_N_con__, self.N_con[1])
		self.assertEqual(self.gwas.gwas2.__current_N__, self.N_cas[1] + self.N_con[1])