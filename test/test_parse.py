from __future__ import division
from ldsc import parse as ps
import unittest
import numpy as np
import nose
from nose_parameterized import parameterized as param


class Test_check_pvalue(unittest.TestCase):
	def setUp(self):
		pass
		
	@nose.tools.raises(ValueError)
	def test_asdf(self):
		pass

	
class Test_check_chisq(unittest.TestCase):
	def setUp(self):
		pass
		
	@nose.tools.raises(ValueError)
	def test_asdf(self):
		pass

	
class Test_check_maf(unittest.TestCase):
	def setUp(self):
		pass
		
	@nose.tools.raises(ValueError)
	def test_asdf(self):
		pass
	

class Test_check_N(unittest.TestCase):
	def setUp(self):
		pass
		
	@nose.tools.raises(ValueError)
	def test_asdf(self):
		pass


class Test_chisq(unittest.TestCase):
	pass
	
	
class Test_betaprod(unittest.TestCase):
	pass
	

class Test_ldscore(unittest.TestCase):
	pass


class Test_ldscore22(unittest.TestCase):
	pass


class Test_M(unittest.TestCase):
	pass


class Test_M22(unittest.TestCase):
	pass