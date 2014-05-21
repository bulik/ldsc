from __future__ import division
from ldsc import parse as ps
import unittest
import numpy as np
import nose
from nose_parameterized import parameterized as param


class Test_dir(unittest.TestCase):

	@nose.tools.raises(ValueError)
	def test(self):
		dir = np.array((1,-1,0))
		ps.check_dir(dir)

 
class Test_check_pvalue(unittest.TestCase):
	
	@nose.tools.raises(ValueError)
	def test_missing(self):
		p = np.array((1,0.5,float('nan')))
		ps.check_pvalue(p)
		
	@nose.tools.raises(ValueError)
	def test_high(self):
		p = np.array((1,0.5,2))
		ps.check_pvalue(p)

	@nose.tools.raises(ValueError)
	def test_low(self):
		p = np.array((1,0.5,0))
		ps.check_pvalue(p)

	
class Test_check_chisq(unittest.TestCase):
	
	@nose.tools.raises(ValueError)
	def test_missing(self):
		chisq = np.array((1,0.5,float('nan')))
		ps.check_chisq(chisq)
		
	@nose.tools.raises(ValueError)
	def test_inf(self):
		chisq = np.array((1,0.5,float('inf')))
		ps.check_chisq(chisq)

	@nose.tools.raises(ValueError)
	def test_inf(self):
		chisq = np.array((1,0.5,-1))
		ps.check_chisq(chisq)


class test_check_maf(unittest.TestCase):
	
	@nose.tools.raises(ValueError)
	def test_missing(self):
		maf = np.array((1,0.5,float('nan')))
		ps.check_maf(maf)
	
	@nose.tools.raises(ValueError)
	def test_high(self):
		maf = np.array((1,0.5,0.2))
		ps.check_maf(maf)
	
	@nose.tools.raises(ValueError)
	def test_low(self):
		maf = np.array((0.1,0.5,0))
		ps.check_maf(maf)


class Test_check_N(unittest.TestCase):

	@nose.tools.raises(ValueError)
	def test_low(self):
		N = np.array((2,1,-1))
		ps.check_N(N)


class Test_chisq(unittest.TestCase):
	
	def test_chisq(self):
		x = ps.chisq('test/parse_test/test.chisq')
		self.assertEqual(list(x['SNP']), ['rs1', 'rs2','rs3'])	
		self.assertEqual(list(x['N']), [100, 100, 100])	
		self.assertEqual(list(x['INFO']), [1, 1, 1])	
		assert np.all(np.abs(x['MAF'] - [0.5, 0.01, 0.01]) < 10e-6)
		assert np.all(x.columns == ['SNP','N','CHISQ','INFO','MAF'])

class Test_betaprod(unittest.TestCase):

	def test_betaprod(self):
		x = ps.betaprod('test/parse_test/test.betaprod')
		self.assertEqual(list(x['SNP']), ['rs1', 'rs2','rs3'])	
		self.assertEqual(list(x['N1']), [100, 100, 100])	
		self.assertEqual(list(x['N2']), [5, 5, 5])	
		self.assertEqual(list(x.columns), ['SNP','N1','BETAHAT1','N2','BETAHAT2'])
		self.assertEqual(list(x['BETAHAT1']), [-1,1,-1])
		self.assertEqual(list(x['BETAHAT2']), [1,-1,1])


class Test_ldscore(unittest.TestCase):

	def test_ldscore(self):
		x = ps.ldscore('test/parse_test/test')
		self.assertEqual(list(x['SNP']), ['rs'+str(i) for i in range(1,23)] ) 
		self.assertEqual(list(x['AL2']), range(1,23) ) 
		self.assertEqual(list(x['BL2']), range(2,46,2) ) 

class Test_ldscore22(unittest.TestCase):
	
	def test_ldscore22(self):
		x = ps.ldscore22('test/parse_test/test')
		self.assertEqual(list(x['SNP']), ['rs'+str(i) for i in range(1,23)] ) 
		self.assertEqual(list(x['AL2']), range(1,23) ) 
		self.assertEqual(list(x['BL2']), range(2,46,2) ) 
	

class Test_M(unittest.TestCase):
	
	@nose.tools.raises(ValueError)
	def test_bad_M(self):
		x = ps.M('test/parse_test/test_bad')
	
	def test_M(self):
		x = ps.M('test/parse_test/test')
		self.assertEqual(list(x), [1000,2000,3000]) 


class Test_M22(unittest.TestCase):

	def test_M22(self):
		x = ps.M22('test/parse_test/test')
		self.assertEqual(list(x), [253, 506])
	