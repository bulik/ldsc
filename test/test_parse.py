from __future__ import division
from ldsc import parse as ps
import unittest
import numpy as np
import nose
from nose_parameterized import parameterized as param


class Test_check_pvalue(unittest.TestCase):
	def setUp(self):
		pass
		
	def test_asdf(self):
		pass

	
class Test_check_chisq(unittest.TestCase):
	def setUp(self):
		pass
		
	def test_asdf(self):
		pass

	
class Test_check_maf(unittest.TestCase):
	def setUp(self):
		pass
		
	def test_asdf(self):
		pass
	

class Test_check_N(unittest.TestCase):
	def setUp(self):
		pass
		
	def test_asdf(self):
		pass


class Test_chisq(unittest.TestCase):
		pass
	
	
class Test_betaprod(unittest.TestCase):
	pass
	

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
	