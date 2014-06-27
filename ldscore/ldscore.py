from __future__ import division
import numpy as np
import bitarray as ba
#import progressbar as pb
import pandas as pd


def getBlockLefts(coords, max_dist):
	'''
	Parameters
	----------
	coords : array
		Array of coordinates. Must be sorted.
	max_dist : float
		Maximum distance between SNPs included in the same window.
	
	Returns
	-------
	block_left : 1D np.ndarray with same length as block_left
		block_left[j] :=  min{k | dist(j, k) < max_dist}. 

	'''
	M = len(coords)
	j = 0
	block_left = np.zeros(M)
	for i in xrange(M):
		while j < M and abs(coords[j] - coords[i]) > max_dist:
			j += 1
		
		block_left[i] = j

	return block_left
	
def block_left_to_right(block_left):
	'''
	Parameters
	----------
	block_left : array
		Array of block lefts.
	
	Returns
	-------
	block_right : 1D np.ndarray with same length as block_left
		block_right[j] := max {k | block_left[k] <= j}
	
	'''
	M = len(block_left)
	j = 0
	block_right = np.zeros(M)
	for i in xrange(M):
		while j < M and block_left[j] <= i:
			j += 1
		
		block_right[i] = j

	return block_right

	
class __GenotypeArrayInMemory__(object):
	'''
	Parent class for various classes containing interfaces for files with genotype 
	matrices, e.g., plink .bed files, etc
	'''
	def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
		self.m = len(snp_list.IDList)
		self.n = n
		self.keep_snps = keep_snps
		self.keep_indivs = keep_indivs
		self.df = np.array(snp_list.df[['CHR','SNP','BP','CM']])
		self.colnames = ['CHR','SNP','BP','CM']
		self.mafMin = mafMin if mafMin is not None else 0
		self._currentSNP = 0
		(self.nru, self.geno) = self.__read__(fname, self.m, n)
		# filter individuals
		if keep_indivs is not None:
			keep_indivs = np.array(keep_indivs, dtype='int')
			if np.any(keep_indivs > self.n):
				raise ValueError('keep_indivs indices out of bounds')
			
			(self.geno, self.m, self.n) = self.__filter_indivs__(self.geno, keep_indivs, self.m,
				self.n)
		
			if self.n > 0:
				print 'After filtering, {n} individuals remain'.format(n=self.n)
			else:
				raise ValueError('After filtering, no individuals remain')
		
		# filter SNPs
		if keep_snps is not None:
			keep_snps = np.array(keep_snps, dtype='int')
			if np.any(keep_snps > self.m): # if keep_snps is None, this returns False
				raise ValueError('keep_snps indices out of bounds')

		(self.geno, self.m, self.n, self.kept_snps, self.freq) = self.__filter_snps_maf__(
			self.geno, self.m, self.n, self.mafMin, keep_snps)
		
		if self.m > 0:
			print 'After filtering, {m} SNPs remain'.format(m=self.m)
		else:
			raise ValueError('After filtering, no SNPs remain')
		
		self.df = self.df[self.kept_snps,:]	
		self.maf = np.minimum(self.freq, np.ones(self.m)-self.freq)
		self.sqrtpq = np.sqrt(self.freq*(np.ones(self.m)-self.freq))
		self.df = np.c_[self.df, self.maf]
		self.colnames.append('MAF')

	def __read__(self,fname, m, n):
		raise NotImplementedError  
	
	def __filter_indivs__(geno, keep_indivs, m, n):
		raise NotImplementedError

	def __filter_maf_(geno, m, n, maf):
		raise NotImplementedError
	'''
	def __ldscore_pbar__(self,m):
		widgets = ['Computed LD for ', pb.Counter(), ' / {m} SNPs ('.format(m=m),
			pb.Timer(), ')']
		pbar = pb.ProgressBar(widgets=widgets,maxval=m).start()
		return pbar
			
	def __filter_indivs_pbar__(self, n):
		widgets = ['Filtered ', pb.Counter(), ' / {n} individuals ('.format(n=n),
			pb.Timer(), ')']
		pbar = pb.ProgressBar(widgets=widgets,maxval=n).start()
		return pbar
		
	def __filter_snps_pbar__(self, m):
		widgets = ['Computed MAF for ', pb.Counter(), ' / {m} SNPs ('.format(m=m),
			pb.Timer(), ')']
		pbar = pb.ProgressBar(widgets=widgets,maxval=m).start()
		return pbar
	'''
	
	### L1 w.r.t. minor allele
	def l1VarBlocks(self, block_left, c, annot=None):
		'''
		Estimates
		
		L1hat(j) := sum_k r_jk
		
		for j = 1,..., M, where the sign of r_jk reflects the correlation between the
		(in-sample) minor alleles at j and k.
		
		'''
		snp_getter = lambda n: self.nextSNPs(n, minorRef=True)
		func = lambda x: x
		l1hat = self.__corSumVarBlocks__(block_left,c,func,snp_getter,annot)
		return l1hat
		
	def l1BlockJackknife(self, block_left, c, annot=None, jN=10):
		func = lambda x: x
		snp_getter = lambda n: self.nextSNPs(n, minorRef=True)
		return self.__corSumBlockJackknife__(block_left, c, func, snp_getter, annot, jN)


	### L1 ^ 2 (unbiased) w.r.t. minor allele 
	def l1sqVarBlocks(self, block_left, c, annot=None):
		'''
		For each SNP j = 1, ... , M computes an approximately unbiased estimate of L1(j)^2
		
		L1hat(j)^2 := ((N-1) / N) ( sum_k rhat_jk ) ^ 2 - ( sum_k L1hat(k) ) / N,
		
		where the sums are taken over all SNPs k within 1cM of j. This would be an unbiased
		estimator if genotypes were multivariate normal. 
		
		'''
		block_right = block_left_to_right(block_left)
		l1hat = self.l1VarBlocks(block_left, c, annot)
		l1sqhat = self.__l1_to_l1sqhat_(l1hat, block_left, block_right)
		return l1sqhat
	
	def __l1_to_l1sqhat_(self, l1hat, block_left, block_right):
		l1sqhat = np.square(l1hat)*(self.n - 1) / self.n
		for k in xrange(self.m):
			l1sqhat[block_left[k]:block_right[k],...] -= l1hat[k] / self.n
		
		return l1sqhat
		
	def l1sqBlockJackknife(self, block_left, c, annot=None, jN=10):
		func = lambda x: x
		func2 = lambda x,y : np.square(np.dot(x,y))
		snp_getter = lambda n: self.nextSNPs(n, minorRef=True)
		return self.__corSumBlockJackknife__(block_left, c, func, snp_getter, annot, jN, func2)


	### L2
	def ldScoreVarBlocks(self, block_left, c, annot=None):
		'''
		Computes an unbiased estimate of L2(j) for j=1,..,M
		
		'''
		func = lambda x: self.__l2_unbiased__(x,self.n)
		snp_getter = self.nextSNPs
		return self.__corSumVarBlocks__(block_left, c, func, snp_getter, annot)
		
	def ldScoreBlockJackknife(self, block_left, c, annot=None, jN=10):
		func = lambda x: np.square(x)
		snp_getter = self.nextSNPs
		return self.__corSumBlockJackknife__(block_left, c, func, snp_getter, annot, jN)
		
	def __l2_unbiased__(self, x, n): 
		denom = n-2 if n>2 else n # allow n<2 for testing purposes
		sq = np.square(x)
		return sq - (1-sq) / denom
		
		
	### L4
	def l4VarBlocks(self, block_left, c, annot=None, jN=10):
		'''
		Computes an estimate of L4 that is unbiased for MVN genotypes.
		'''
		func = lambda x: self.__l4_unbiased__(x,self.n)
		snp_getter = self.nextSNPs
		return self.__corSumVarBlocks__(block_left, c, func, snp_getter, annot)
	
	def __l4_unbiased__(self, x, n):
		raise NotImplementedError
	
	def l4BlockJackknife(self, block_left, c, annot=None, jN=10):
		func = lambda x: np.square(np.square((x)))
		snp_getter = self.nextSNPs
		return self.__corSumBlockJackknife__(block_left, c, func, snp_getter, annot, jN)
		
		
	# general methods for calculating sums of Pearson correlation coefficients
	def __corSumVarBlocks__(self, block_left, c, func, snp_getter, annot=None):
		'''
		Parameters
		----------		
		block_left : np.ndarray with shape (M, )
			block_left[i] = index of leftmost SNP included in LD Score of SNP i.
			if c > 1, then only entries that are multiples of c are examined, and it is 
			assumed that block_left[a*c+i] = block_left[a*c], except at 
			the beginning of the chromosome where the 0th SNP is included in the window.

		c : int
			Chunk size.
		func : function
			Function to be applied to the genotype correlation matrix. Before dotting with
			annot. Examples: for biased L2, np.square. For biased L4, 
			lambda x: np.square(np.square(x)). For L1, lambda x: x.
		snp_getter : function(int)
			The method to be used to get the next SNPs (normalized genotypes? Normalized 
			genotypes with the minor allele as reference allele? etc)
		annot: numpy array with shape (m,n_a)
			SNP annotations.
		
		Returns
		-------
		cor_sum : np.ndarray with shape (M, num_annots)
			Estimates.
			
		'''	
		m, n = self.m, self.n
		block_sizes = np.array(np.arange(m) - block_left)
		block_sizes = np.ceil(block_sizes / c)*c
		if not np.any(annot):
			annot = np.ones((m,1))
		else:
			annot_m = annot.shape[0]
			if annot_m != self.m:
				raise ValueError('Incorrect number of SNPs in annot')
			
		n_a = annot.shape[1] # number of annotations
		cor_sum = np.zeros((m,n_a))
		# b = index of first SNP for which SNP 0 is not included in LD Score
		b = np.nonzero(block_left > 0)
		if np.any(b):
			b = b[0][0]
		else:
			b = m
		b = int(np.ceil(b/c)*c) # round up to a multiple of c
		if b > m:
			c = 1; b = m
		l_A = 0; nsq = n**2 # l_A := index of leftmost SNP in matrix A
		
		A = snp_getter(b)
		rfuncAB = np.zeros((b,c))
		rfuncBB = np.zeros((c,c))
		# chunk inside of block
		#pbar = self.__ldscore_pbar__(m)

		for l_B in xrange(0,b,c): # l_B := index of leftmost SNP in matrix B
			#pbar.update(l_B)
			B = A[:,l_B:l_B+c]
			np.dot(A.T, B / n, out=rfuncAB)
			rfuncAB = func(rfuncAB)
			cor_sum[l_A:l_A+b,:] += np.dot(rfuncAB,annot[l_B:l_B+c,:])
		
		# chunk to right of block
		b0 = b
		md = int(c*np.floor(m/c))
		end = md + 1 if md !=m else md
		for l_B in xrange(b0,end,c):
			#pbar.update(l_B)
			# update the block
			old_b = b
			b = block_sizes[l_B]
			if l_B > b0 and b > 0:
				# block_size can't increase more than c	
				# block_size can't be less than c unless it is zero
				# both of these things make sense
				A = np.hstack((A[:,old_b-b+c:old_b],B))
				l_A += old_b-b+c
			elif l_B == b0 and b > 0:
				A = A[:,b0-b:b0]
				l_A = b0-b
			elif b == 0: # no SNPs to left in window, e.g., after a sequence gap
				A = np.array(()).reshape((n,0))
				l_A = l_B
			if l_B == md: 
				c = m - md
				rfuncAB = np.zeros((b,c))
				rfuncBB = np.zeros((c,c))
			if b != old_b:
				rfuncAB = np.zeros((b,c))
			
			B = snp_getter(c) 		
			np.dot(A.T, B / n, out=rfuncAB) 
			rfuncAB = func(rfuncAB)
			cor_sum[l_A:l_A+b,:] += np.dot(rfuncAB, annot[l_B:l_B+c,:])
			cor_sum[l_B:l_B+c,:] += np.dot(annot[l_A:l_A+b,:].T,rfuncAB).T
			np.dot(B.T, B / n, out=rfuncBB)
			rfuncBB = func(rfuncBB)
			cor_sum[l_B:l_B+c,:] += np.dot(rfuncBB, annot[l_B:l_B+c,:])
		
		#pbar.finish()
		return cor_sum
	
# 	def __corSumBlockJackknife__(self, block_left, c, func1, snp_getter, annot=None, jN=10,
# 		func2=np.dot):
# 		M, N = self.m, self.n; 
# 		if jN > N:
# 			raise ValueError('jN must be <= N')
# 		jSize = int(np.ceil(N/jN))
# 		last_jSize = N - (jN-1)	*jSize
# 		def jsize(i):
# 			if i < jN: return jSize
# 			elif i == jN: return last_jSize
# 			else: raise ValueError
# 
# 		block_sizes = np.array(np.arange(M) - block_left)
# 		block_sizes = np.ceil(block_sizes / c)*c
# 		if not np.any(annot):
# 			N_A = 1 # number of annotations
# 			annot = np.ones((M,N_A))
# 		else:
# 			N_A = annot.shape[1]
# 			annot_m = annot.shape[0]
# 			if annot_m != self.m:
# 				raise ValueError('Incorrect number of SNPs in annot')
# 				
# 		# b = index of first SNP for which SNP 0 is not included in LD Score
# 		b = np.nonzero(block_left > 0)
# 		if np.any(b):
# 			b = b[0][0]
# 		else:
# 			b = M
# 		b = int(np.ceil(b/c)*c) # round up to a multiple of c
# 		if b > M:
# 			c = 1; b = M
# 		l_A = 0; nsq = N**2 # l_A := index of leftmost SNP in matrix A
# 		A = snp_getter(b) # read first b0 SNPs into the block 
# 		rAB = np.zeros((b,c,jN))
# 		rBB = np.zeros((c,c,jN))
# 		MD = int(c*np.floor(M/c))
# 		# num SNPs x num annotations x num jknife blocks + 1 
# 		# last index in 3rd axis is LD Score with nothing deleted
# 		cor_sum = np.zeros((M,N_A,jN+1)) 
# 		# chunk inside of block
# 		pbar = self.__ldscore_pbar__(M)
# 		for l_B in xrange(0,b,c): # l_B := index of leftmost SNP in matrix B
# 			pbar.update(l_B)
# 			B = A[:,l_B:l_B+c]		
# 			for ji in xrange(0, jN):
# 				i = ji*jSize
# 				jA = A[i:i+jsize(ji),:]
# 				jB = B[i:i+jsize(ji),:]
# 				rAB[:,:,ji] = np.dot(jA.T, jB) / jsize(ji)
# 			for j in xrange(0,jN+1):
# 				ii = np.arange(jN) != j
# 				jRfuncAB = func1(np.mean(rAB[:,:,ii], axis=2)) 
# 				cor_sum[l_A:l_A+b,:,j] += func2(jRfuncAB, annot[l_B:l_B+c,:])
# 
# 		# chunk to right of block
# 		b0 = b
# 		MD = int(c*np.floor(M/c))
# 		end = MD + 1 if MD !=M else MD
# 		for l_B in xrange(b0, end, c):
# 			pbar.update(l_B)
# 			old_b = b
# 			b = block_sizes[l_B]
# 			
# 			if l_B > b0 and b > 0:
# 				A = np.hstack((A[:,old_b-b+c:old_b],B)) # block_size can't increase more than c	
# 				l_A += old_b-b+c
# 			elif l_B == b0 and b > 0:
# 				A = A[:,b0-b:b0]
# 				l_A = b0-b
# 			elif b == 0: # no SNPs to left in window, e.g., after a sequence gap
# 				A = np.array(()).reshape((N,0))
# 				l_A = l_B
# 			
# 			if l_B == MD:
# 				c = M - MD
# 				#rAB = np.zeros((b,c,jN))
# 				#rBB = np.zeros((c,c,jN))
# 			#if b != old_b:
# 			#rAB = np.zeros((b,c,jN))
# 			rAB = []; rBB = []
# 			
# 			B = snp_getter(c)
# 			
# 			# compute block values
# 			for ji in xrange(0, jN):
# 				i = ji*jSize
# 				jA = A[i:i+jsize(ji),:]
# 				jB = B[i:i+jsize(ji),:]
# 				rAB.append(np.dot(jA.T, jB) / jsize(ji))
# 				rBB.append(np.dot(jB.T, jB) / jsize(ji))
# 				#rAB[:,:,ji] = np.dot(jA.T, jB) / jsize(ji)
# 				#rBB[:,:,ji] = np.dot(jB.T,jB) / jsize(ji)
# 			
# 			rAB.append(np.zeros(rAB[1].shape))
# 			rBB.append(np.zeros(rBB[1].shape))
# 			sum_rAB = np.sum(rAB, axis=0)
# 			sum_rBB = np.sum(rBB, axis=0)
# 
# 			# compute delete-k values
# 			for j in xrange(0,jN+1):
# 				#ii = np.arange(jN) != j
# 				#jRfuncAB = func1(np.mean(rAB[:,:,ii], axis=2))  
# 				#jRfuncBB = func1(np.mean(rBB[:,:,ii], axis=2))
# 				denom = jN if j != jN+1 else jN+1
# 				jRfuncAB = func1( (sum_rAB - rAB[j]) / denom )
# 				jRfuncBB = func1( (sum_rBB - rBB[j]) / denom )
# 				cor_sum[l_A:l_A+b,:,j] += func2(jRfuncAB, annot[l_B:l_B+c,:])
# 				cor_sum[l_B:l_B+c,:,j] += func2(annot[l_A:l_A+b,:].T,jRfuncAB).T
# 				cor_sum[l_B:l_B+c,:,j] += func2(jRfuncBB, annot[l_B:l_B+c,:])
# 				
# 		# compute jackknife SE and jackknife estimate
# 		cor_sum_se = np.sqrt(jN-1)*np.std(cor_sum[:,:,0:jN],axis=2)	
# 		cor_sum_jn = jN*cor_sum[:,:,jN] - (jN-1)*np.mean(cor_sum[:,:,0:jN], axis=2)
# 		pbar.finish()
# 		print 
# 		return (cor_sum_jn, cor_sum_se)			


class PlinkBEDFile(__GenotypeArrayInMemory__):
	'''
	Interface for Plink .bed format 
	'''
	def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
		self._bedcode = {
			2: ba.bitarray('11'), 
			9: ba.bitarray('10'), 
			1: ba.bitarray('01'), 
			0:ba.bitarray('00')
			}
		
		__GenotypeArrayInMemory__.__init__(self, fname, n, snp_list, keep_snps=keep_snps, 
			keep_indivs=keep_indivs, mafMin=mafMin) 
	
	def __read__(self, fname, m, n):
		if not fname.endswith('.bed'):
			raise ValueError('.bed filename must end in .bed')

		fh = open(fname,'rb')
		magicNumber = ba.bitarray(endian="little")
		magicNumber.fromfile(fh, 2)
		bedMode = ba.bitarray(endian="little")
		bedMode.fromfile(fh, 1)
		### TODO this is ugly
		e = (4 - n % 4) if n % 4 != 0 else 0
		nru = n + e
		self.nru = nru
		# check magic number
		if magicNumber != ba.bitarray('0011011011011000'):
			raise IOError("Magic number from Plink .bed file not recognized")
			
		if bedMode != ba.bitarray('10000000'):
			raise IOError("Plink .bed file must be in default SNP-major mode")	

		# check file length
		self.geno = ba.bitarray(endian="little")
		self.geno.fromfile(fh)
		self.__test_length__(self.geno,self.m,self.nru)
		return (self.nru, self.geno)
		
	def __test_length__(self, geno, m, nru):
		exp_len = 2*m*nru
		real_len = len(geno)
		if real_len != exp_len:
			s = "Plink .bed file has {n1} bits, expected {n2}"
			raise IOError(s.format(n1=real_len, n2=exp_len))
	
	def __filter_indivs__(self, geno, keep_indivs, m, n):
		n_new = len(keep_indivs)
		e = (4 - n_new % 4) if n_new % 4 != 0 else 0
		nru_new = n_new + e
		nru = self.nru
		z = ba.bitarray(m*2*nru_new, endian="little")
		#pbar = self.__filter_indivs_pbar__(len(keep_indivs))
		for e, i in enumerate(keep_indivs):
			#pbar.update(e)
			z[2*e::2*nru_new] = geno[2*i::2*nru]
			z[2*e+1::2*nru_new] = geno[2*i+1::2*nru]

		#pbar.finish()
		self.nru = nru_new
		return (z, m, n_new)
	
	def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
		'''
		Credit to Chris Chang and the Plink2 developers for this algorithm
		Modified from plink_filter.c
		https://github.com/chrchang/plink-ng/blob/master/plink_filter.c
				
		Genotypes are read forwards (since we are cheating and using endian="little")
		
		A := (genotype) & 1010...
		B := (genotype) & 0101...
		C := (A >> 1) & B
		
		Then 
		
		a := A.count() = missing ct + hom major ct
		b := B.count() = het ct + hom major ct
		c := C.count() = hom major ct
		
		Which implies that
		
		missing ct = a - c
		# of indivs with nonmissing genotype = n - a + c
		major allele ct = b + c
		major allele frequency = (b+c)/(2*(n-a+c))
		het ct + missing ct = a + b - 2*c
		
		Why does bitarray not have >> ????
		
		'''
	
		nru = self.nru
 		m_poly = 0
 		y = ba.bitarray()
 		if keep_snps is None: keep_snps = xrange(m)
 		kept_snps = []; freq = []
 		#pbar = self.__filter_snps_pbar__(len(keep_snps))
		for e,j in enumerate(keep_snps):
			#pbar.update(e)
			z = geno[2*nru*j:2*nru*(j+1)]
			A = z[0::2]; a = A.count()
			B = z[1::2]; b = B.count()
			c = (A&B).count()
			major_ct = b + c # number of copies of the major allele
			n_nomiss = n - a + c # number of individuals with nonmissing genotypes
			f = major_ct / (2*n_nomiss) if n_nomiss > 0 else 0
			het_miss_ct = a+b-2*c # remove SNPs that are only either het or missing
			if np.minimum(f,1-f) > mafMin and het_miss_ct < n:
				freq.append(f)
				y += z
				m_poly += 1			
				kept_snps.append(j)	
		
		#pbar.finish()	
 		return (y, m_poly, n, kept_snps, freq)
		
	def nextSNPs(self, b, minorRef=None):
		'''
		return an n x b matrix of the next b SNPs
		'''
		try:
			b = int(b)
			if b <= 0:
				raise ValueError("b must be > 0")
		except TypeError:
			raise TypeError("b must be an integer")
		
		if self._currentSNP + b > self.m:
			s = '{b} SNPs requested, {k} SNPs remain'
			raise ValueError(s.format(b=b, k=(self.m-self._currentSNP)))

		c = self._currentSNP
		n = self.n
		nru = self.nru
		slice = self.geno[2*c*nru:2*(c+b)*nru]
		X = np.array(slice.decode(self._bedcode), dtype="float64").reshape((b,nru)).T
		X = X[0:n,:]
		Y = np.zeros(X.shape)
		for j in xrange(0,b):
			newsnp = X[:,j]
			ii = newsnp != 9
			avg = np.mean(newsnp[ii])
			newsnp[np.logical_not(ii)] = avg
			denom = np.std(newsnp)
			if denom == 0:
				denom = 1
			
			if minorRef is not None and self.freq[self._currentSNP + j] > 0.5:
				denom = denom*-1
				
			Y[:,j] = (newsnp - avg)/denom

		self._currentSNP += b
		return Y 
		
		
class VcfBINFile(__GenotypeArrayInMemory__):
	'''
	Interface for genotypes stored as a python bitarray
	'''
	def __read__(self, fname, m, n):
		if not fname.endswith('.bin'):
			raise ValueError('.bin filename must end in .bin')

		n, m = self.n, self.m
		geno = ba.bitarray(endian="big")
		magicNumber = ba.bitarray(endian="big")
		fh = open(fname,'rb')
		magicNumber.fromfile(fh, 4)
		ldsc = ba.bitarray()
		ldsc.fromstring('ldsc')
		if magicNumber != ldsc:
			raise IOError("Magic number from .bin file not recognized")
		# check file length
		geno.fromfile(fh)
		exp_len = m*n + ((8-m*n) % 8)
		real_len = len(geno)
		if real_len != exp_len:
			s = "Binary .bin file has {n1} bits, expected {n2}"
			raise IOError(s.format(n1=real_len, n2=exp_len))

		return (n, geno)
		
	def __filter_indivs__(self, geno, keep_indivs, m, n):
		n_new = len(keep_indivs)
		z = ba.bitarray(m*n_new, endian="big")
		#pbar = self.__filter_indivs_pbar__(len(keep_indivs))
		for e, i in enumerate(keep_indivs):
			#pbar.update(e)
			z[e::n_new] = geno[i:m*n:n]

		#pbar.finish()
		return (z, m, n_new)

	def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
		y = ba.bitarray(m*n, endian="big")
		m_poly = 0; freq =  []
		if keep_snps is None: keep_snps = xrange(m)
		kept_snps = []
 		#pbar = self.__filter_snps_pbar__(len(keep_snps))
		for e,j in enumerate(keep_snps):
			#pbar.update(e)
			z = geno[j*n:(j+1)*n]
			c = z.count()
			if c < n and np.minimum(c/n,1-c/n) > mafMin:
				y[m_poly*n:(m_poly+1)*n] = z
				m_poly += 1
				freq.append(c/n)
				kept_snps.append(j)
				
		#pbar.finish()
		return (y[0:m_poly*n], m_poly, n, kept_snps, freq)
		
	def nextSNPs(self, b, minorRef=None):
		'''
		returns an n x b matrix of normalized genotypes for the next b SNPs
		'''
		b = int(b)
		if b <= 0:
			raise ValueError("b must be > 0")
		
		if self._currentSNP + b > self.m:
			s = '{b} SNPs requested, {k} SNPs remain'
			raise ValueError(s.format(b=b, k=(self.m-self._currentSNP)))
		
		b = min(self.m - self._currentSNP, b)
		n = self.n
		c = self._currentSNP
		X = np.array( self.geno[c*n:(c+b)*n].tolist(), dtype="float64").reshape((b,n)).T
		if minorRef is not None:
			flip_ref = np.ones(b)
			flip_ref[np.nonzero(np.array(self.freq[c:(c+b)]) > 0.5)] = -1
			denom = self.sqrtpq[c:c+b] * flip_ref
			X = (X - self.freq[c:c+b] ) / denom
		else:
			X = (X - self.freq[c:c+b] ) / self.sqrtpq[c:c+b]
		
		self._currentSNP += b
		return X		


class AnnotFile():

	def __init__(self, fname):
		self._read(fname)

	def _read(self, fname):
		dtype_dict = {0:"str", 1:"float64", 2:"string"}
		self.annot = pd.read_csv(fname, header=0, delim_whitespace=True, dtype=dtype_dict)
		numAnnots = self.annot.shape[1] - 3
		self.annotIDs = self.annot['SNP']
		self.annotMatrix = np.array(self.annot.ix[:,4:], dtype="float64")