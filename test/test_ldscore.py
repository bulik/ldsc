import ldsc.ldscore as ld
import unittest
import bitarray as ba
import numpy as np
import nose
import pandas as pd
from nose_parameterized import parameterized as param


##########################################################################################
#                                  MISC FUNCTIONS                                        #
##########################################################################################


@param([
	(np.arange(1,6),5,np.zeros(5)),
	(np.arange(1,6),0,np.arange(0,5)),
	((1,4,6,7,7,8),2,(0,1,1,2,2,2))])
def test_getBlockLefts(coords, max_dist, correct):
	assert np.all(ld.getBlockLefts(coords, max_dist) == correct)

@param([
	((0,0,0,0,0),(5,5,5,5,5)),
	((0,1,2,3,4,5), (1,2,3,4,5,6)),
	((0,0,2,2), (2,2,4,4))
	])
def test_block_left_to_right(block_left, correct_answer):
	block_right = ld.block_left_to_right(block_left)
	print block_right - correct_answer
	assert np.all(block_right == correct_answer)


##########################################################################################
#                                    BED PARSER                                          #
##########################################################################################


class test_bed(unittest.TestCase):
 
 	def setUp(self):
 		self.M = 8
 		self.N = 5
		self.bim = ld.PlinkBIMFile('test/plink_test/plink.bim')

 	def test_bed(self):
 		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim)
 		# remove three monomorphic SNPs
 		print bed.geno
		print bed.m
 		assert bed.m == 4
 		# no individuals removed	
 		print bed.n	
 		assert self.N == bed.n
 		# 5 indivs * 4 polymorphic SNPs
 		print len(bed.geno)
 		assert len(bed.geno) == 64
 		print bed.freq
 		correct = np.array([0.59999999999999998, 0.59999999999999998, 0.625, 0.625])
 		assert np.all(bed.freq == correct)
 				
 	def test_filter_snps(self):
 		keep_snps = [1,4]
 		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim,
 			keep_snps=keep_snps)
 		assert bed.m == 1
 		assert bed.n == 5
 		assert bed.geno == ba.bitarray('0001011111000000')
 		
 	def test_filter_indivs(self):
 		keep_indivs = [0,1]
 		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim, 
 			keep_indivs=keep_indivs)
 		print bed.m, bed.n
 		print bed.geno
 		assert bed.m == 2
 		assert bed.n == 2
 		assert bed.geno == ba.bitarray('0001000000010000')
 		
	def test_filter_indivs_and_snps(self):
		keep_indivs = [0,1]
		keep_snps = [1,5]
		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim, 
			keep_snps=keep_snps, keep_indivs=keep_indivs)
		assert bed.m == 1
		assert bed.n == 2
		print bed.geno
		assert bed.geno == ba.bitarray('00010000')
	
	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		bed = ld.PlinkBEDFile('test/plink_test/plink.bim', 9, self.bim)
	
	@nose.tools.raises(ValueError)	
	def test_nextSNPs_errors1(self):
		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim)
		bed.nextSNPs(0)
	
	@nose.tools.raises(ValueError)	
	def test_nextSNPs_errors2(self):
		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim)
		bed.nextSNPs(5)
				
	@param.expand([([1]),([2]),([3])])
	def test_nextSNPs(self, b):
		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim)
		x = bed.nextSNPs(b)
		print x
		print np.std(x,axis=0)
		assert x.shape == (5,b)
		assert np.all(np.abs(np.mean(x, axis=0)) < 0.01)
		assert np.all(np.abs(np.std(x, axis=0) - 1) < 0.01)		

	def test_nextSNPs_maf_ref(self):
		b = 4
		bed = ld.PlinkBEDFile('test/plink_test/plink.bed', self.N, self.bim)
		x = bed.nextSNPs(b)
		bed._currentSNP -= b
		y = bed.nextSNPs(b, minorRef=True)
		assert np.all(x == -y)

##########################################################################################
#                                     VCF PARSER                                         #
##########################################################################################


class test_bin(unittest.TestCase):
	
	def setUp(self):
		self.M = 6
		self.N = 5
		self.geno = ba.bitarray('01100011000101001010')
		self.snp = ld.VcfSNPFile('test/vcf_test/test.snp')
		'''
		The genotype matrix looks like this before any filtering
		
		(indivs) x (snps)
		
		010000
		011111
		011100
		010011
		010000
		 		
		The two monomorphic SNPs at left should get removed in all tests
		'''
		
	def test_bin(self):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp, mafMin=0.2)
		print bin.geno
		print len(bin.geno)
		print bin.freq
		
		assert bin.m == 4
		assert bin.n == self.N
		assert len(bin.geno) == bin.n*bin.m
		assert bin.geno == self.geno
		assert np.all(bin.freq==0.4*np.ones(4))

	@nose.tools.raises(ValueError)	
	def test_mafMin(self):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp, mafMin=0.45)
		
	def test_filter_snps(self):
		'''
		The genotype matrix should look like this after filtering SNPs
		
		100
		111
		110
		101
		100
		
		The monomorphic SNP at left should be removed by _filter_monomorphic
		'''		
		k = [1,3,4]
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp, keep_snps=k)
		print bin.geno
		assert bin.m == 2
		assert bin.n == 5
		assert bin.geno == ba.bitarray('0110001010')
	
	def test_filter_indivs(self):
		'''
		The genotype matrix should look like this after filtering individuals
		
		011111
		011100
		010011	
		
		The monomorphic SNPs at left should be removed by _filter_monomorphic
		'''
		k = [1,2,3]
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp, keep_indivs=k)
		assert bin.n == 3
		assert bin.m == 4 
		assert bin.geno == ba.bitarray('110110101101')

	def test_filter_snps_and_indivs(self):
		'''
		The genotype matrix should look like this after filtering SNPs and individuals
		
		111
		110
		101
		
		The monomorphic SNP at left should be removed by _filter_monomorphic
		'''		
		ks = [1,3,5]
		ki = [1,2,3]
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp, keep_snps=ks, 
			keep_indivs=ki)
		assert bin.n == 3
		assert bin.m == 2 
		assert bin.geno == ba.bitarray('110101')
		
	@nose.tools.raises(ValueError)
	def test_bad_filename(self):
		bin = ld.VcfBINFile('test/vcf_test/test.ind', self.N, self.snp)

	@nose.tools.raises(ValueError)	
	def test_nextSNPs_errors1(self):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp)
		bin.nextSNPs(0)
	
	@nose.tools.raises(ValueError)	
	def test_nextSNPs_errors2(self):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp)
		bin.nextSNPs(7)
	
	@param.expand([([1]),([2]),([3])])
	def test_nextSNPs(self, b):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp)
		x = bin.nextSNPs(b)
		assert x.shape == (5,b)
		print np.mean(x,axis=0)
		# roundoff error --> can't test equality to zero directly
		assert np.all(np.abs(np.mean(x, axis=0)) < 0.01)
		assert np.all(np.abs(np.std(x, axis=0) - 1) < 0.01)
	
	@param.expand([([1]),([2]),([3])])
	def test_nextSNPs_maf_ref(self, b):
		bin = ld.VcfBINFile('test/vcf_test/test.bin', self.N, self.snp)
		x = bin.nextSNPs(b)
		bin._currentSNP -= b
		y = bin.nextSNPs(b, minorRef=True)
		assert np.all(x == y)
		# switch reference alleles
		bin.geno = ~ bin.geno; bin.freq = np.ones(bin.m) - bin.freq
		print bin.geno
		bin._currentSNP -=b
		z = bin.nextSNPs(b, minorRef=True)
		assert np.all(y == z)


##########################################################################################
#                                       LD SCORE                                         #
##########################################################################################


class test_ldscore_vcf(unittest.TestCase):
 
 	def setUp(self):
		self.M = 10
		self.N = 2
		annot = np.array((1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1),dtype='float64')
		self.annot = annot.reshape((10,2))
		self.snp = ld.VcfSNPFile('test/vcf_test/test.ldscore.snp')
		self.bin = ld.VcfBINFile('test/vcf_test/test.ldscore.bin', self.N, self.snp)
		
	def test_no_snps_to_left(self):
		# test the edge case where no SNPs to left are in the window
		block_left = np.arange(10)
		x = self.bin.ldScoreVarBlocks(block_left,1,annot=self.annot)
		assert np.all(x == self.annot)

	def test_no_annot(self):
		block_left = np.array((0,0,0,0,2,2,4,4,8,8))
		x = self.bin.ldScoreVarBlocks(block_left,2,annot=None).reshape((10))
		correct_x = np.array((4,4,6,6,6,6,4,4,2,2))
		assert np.all(x == correct_x)

	@param.expand([(1,(2,2,2,2,2,4,2,4,2,4,2,4,1,3,1,3,1,1,1,1)),
		(2,(2,2,2,2,2,4,2,4,2,4,2,4,1,3,1,3,1,1,1,1)),
		(3,(2,4,2,4,2,4,4,5,4,5,4,5,3,4,3,4,3,4,2,2))])
	def test_var_block_size(self, c, correct_x):
		correct_x = np.array(correct_x,dtype='float64').reshape((10,2))
		block_left = np.array((0,0,0,0,2,2,4,4,8,8))
		x = self.bin.ldScoreVarBlocks(block_left,c,self.annot)
		print self.annot
		print x - correct_x
		assert np.all(x == correct_x)
	
	def test_l1_to_l1sqhat_(self):
		block_left = np.zeros(10)
		block_right = np.repeat(10,10)
		l1hat = np.ones((10,2))
		x = self.bin._GenotypeArrayInMemory____l1_to_l1sqhat_(l1hat, block_left, 
			block_right)
		assert np.all(x == -4.5)
		
	
##########################################################################################
#                               LD SCORE + JACKKNIFE SE                                  #
##########################################################################################
	
class test_ldscore_jackknife_vcf(unittest.TestCase):
 
 	def setUp(self):
		self.M = 10
		self.N = 2
		annot = np.array((1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1),dtype='float64')
		self.annot = annot.reshape((10,2))		
		self.snp = ld.VcfSNPFile('test/vcf_test/test.ldscore.snp')
		self.bin = ld.VcfBINFile('test/vcf_test/test.ldscore.bin', self.N, self.snp)
	
	def test(self):
		block_left = np.array((0,0,1,1,1,1,4,4,4,4))
		x = self.bin.ldScoreBlockJackknife(block_left,c=1,annot=None,jN=2)
	
	@param.expand([(1,(2,2,2,2,2,4,2,4,2,4,2,4,1,3,1,3,1,1,1,1)),
		(2,(2,2,2,2,2,4,2,4,2,4,2,4,1,3,1,3,1,1,1,1)),
		#(3,(2,4,2,4,2,4,4,5,4,5,4,5,3,4,3,4,3,4,2,2))
		])
	def test_var_block_size_jknife(self, c, correct_x):
		correct_x = np.array(correct_x,dtype='float64').reshape((10,2))
		block_left = np.array((0,0,0,0,2,2,4,4,8,8))
		x = self.bin.ldScoreBlockJackknife(block_left,c=c,annot=self.annot,jN=2)[0]
		assert np.all(x == correct_x)