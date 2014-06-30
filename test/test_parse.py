from __future__ import division
from ldscore import parse as ps
import unittest
import numpy as np
import pandas as pd
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
	
	def test_ldscore_loop(self):
		x = ps.ldscore('test/parse_test/test',2)
		self.assertEqual(list(x['SNP']), ['rs'+str(i) for i in range(1,3)] ) 
		self.assertEqual(list(x['AL2']), range(1,3) ) 
		self.assertEqual(list(x['BL2']), range(2,6,2) ) 
	

class Test_M(unittest.TestCase):
	
	@nose.tools.raises(ValueError)
	def test_bad_M(self):
		x = ps.M('test/parse_test/test_bad')
	
	def test_M(self):
		x = ps.M('test/parse_test/test')
		self.assertEqual(list(x), [1000,2000,3000]) 


class Test_M_loop(unittest.TestCase):

	def test_M_loop(self):
		x = ps.M('test/parse_test/test', 2)
		self.assertEqual(list(x), [3, 6])
		
		
class test_snp(unittest.TestCase):

	def test_snp(self):
		snp = ps.VcfSNPFile('test/vcf_test/test.snp')
		print snp.IDList
		assert snp.n == 6
		correct = np.array(['rs1','rs2','rs3','rs4','rs5','rs6'])
		assert np.all(snp.IDList.values.reshape(6) == correct)

	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		ind = ps.VcfSNPFile('test/plink_test/plink.asdf')
		
	def test_loj(self):
		snp = ps.VcfSNPFile('test/vcf_test/test.snp')
		df2 = pd.DataFrame(['rs1','rs4','rs6'])
		ii = snp.loj(df2)
		assert np.all(ii == [0,3,5])
		
		
class test_ind(unittest.TestCase):
	
	def test_ind(self):
		ind = ps.VcfINDFile('test/vcf_test/test.ind')
		print ind.IDList
		assert ind.n == 5
		correct = np.array(['per1', 'per2', 'per3', 'per4', 'per5'])
		assert np.all(ind.IDList.values.reshape(5) == correct)

	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		ind = ps.VcfINDFile('test/plink_test/plink.snp')
		
		
class test_fam(unittest.TestCase):

	def test_fam(self):
		fam = ps.PlinkFAMFile('test/plink_test/plink.fam')
		print fam.IDList
		assert fam.n == 5
		correct = np.array(['per0', 'per1', 'per2', 'per3', 'per4'])
		assert np.all(fam.IDList.values.reshape((5,)) == correct)
		
	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		fam = ps.PlinkFAMFile('test/plink_test/plink.bim')


class test_bim(unittest.TestCase):

	def test_bim(self):
		bim = ps.PlinkBIMFile('test/plink_test/plink.bim')
		print bim.IDList
		assert bim.n == 8
		correct = np.array(['rs_0','rs_1','rs_2','rs_3','rs_4','rs_5','rs_6','rs_7'])
		assert np.all(bim.IDList.values.reshape(8) == correct)
	
	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		bim = ps.PlinkBIMFile('test/plink_test/plink.fam')
	
		
class test_filter_file(unittest.TestCase):

	def test_filter(self):
		filter = ps.FilterFile('test/vcf_test/test.ind')
		print filter.IDList
		assert filter.n == 5
		correct = np.array(['per1', 'per2', 'per3', 'per4', 'per5'])
		assert np.all(filter.IDList.values.reshape(5) == correct)
	