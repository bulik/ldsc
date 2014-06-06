from __future__ import division
import numpy as np
import scipy.stats as spstats
import simulate as sim


class GWAS(object):
	'''
	Parent class for all single GWAS simulation classes. 

	Parameters
	----------
	(for the __init__ method)
	
	M : int
		Number of bi-allelic SNPs sampled.
	N : int
		Number of diploid individuals sampled.
	mafs : np.ndarray of floats with shape (M, ).
		Minor allele frequencies (must be in the interval (0,0.5]
	betas : np.matrix of floats with shape (M, # phenotypes)
		Additive genetic effect sizes, possibly for several phenotypes if several phenotypes
		are needed for ascertainment.
	hsqs : np.matrix of floats with shape (# phenotypes, )
		Heritability of all phenotypes needed for ascertainment. All values must lie in the 
		interval [0,1].
	e_var : np.matrix of floats with shape (# phenotypes, # phenotypes), optional
		Environmental covariance matrix. Required if # of phenotypes > 1. If 
		# phenotypes == 1, then the environmental variance is set to 1 - hsqs.
	
	Attributes
	----------
	(all parameters of __init__ are stored with the same names)
	
	geno : ndarray of floats with shape (N,M)
		Matrix of normalized genotypes. Accessed with self.__append__
	pheno : matrix of floats with shape (N, # phenotypes)
		Phenotypes. The phenotype used in association tests is in the first column.	
	bv : matrix of floats with shape (N, # phenotypes)  
		Breeding values. Go read some cow genetics papers. 
	ev : matrix of floats with shape (N, # phenotypes) 
		Environmental effects.
	num_phenos : int
		Number of phenotypes. The first phenotype is used for association testing; the rest
		are used for ascertainment.
	__current_N__ : int
		Number of individuals that have been sampled, either by calls to __sample__ or
		__insert__. Set equal to zero by __init__() and modified by each subsequent call to
		__append__().
	__has_sampled__ : bool
		True if all samples have been collected has been called; otherwise false.	Set to False
		by __init__() then modified by __append__().

	Methods
	-------
	
	betahat
		Returns ndarray with shape (M,) consisting of slopes from marginal regression of 
		phenotype on genotype for all M SNPs.
	chisq
		Returns ndarray with shape (M,) consisting of Armitage Trend Test statistics (equal to 
		N * betahat ^ 2 from marginal regression of phenotype on genotype for all M SNPs.
	sample()
		Draws (geno, bv, ev, pheno), where geno = an ndarray of genotypes with shape (N, M);
		bv = an ndarray of breeding values with shape (N, # phenotypes); ev = an ndarray 
		of environmental effects with shape (N, # phenotypes); pheno = an ndarray of shape 
		(N, # phenotypes). Calls __append__(geno, bv, ev, pheno).
	__draw_geno__(num_samples)
		Returns a matrix of normalized genotypes with shape (N, num_samples)
	__append__(geno, bv, ev, pheno)
		Raises ValueError if the number of samples to be appended exceeds self.N or if the 
		dimensions of bv, ev or pheno do not match self.num_phenos. If no error is raised,
		appends geno, bv, ev, pheno to self.geno, self.bv, self.ev, self.pheno
	__bv__(geno, betas)
		Returns breeding values with shape (N, # phenotypes)
	__ev__(num, var)
		Returns environmental effects with shape (N, # phenotypes) 
	__sumstats__(which)
		Calculates and caches betahat or chisq if not already calculated. Returns 
		cached value of betahat or chisq
	__calc_pheno__(bv, ev)
		Calculate phenotype from breeding value and environmental effects. Child classes 
		should override. Raises NotImplementedError
	__insert__(geno, bv, ev)
		Inserts geno, bv, ev, __calc_pheno__(bv, ev) into self.geno, self.bv, self.ev and 
		self.pheno respectively after performing some input checking.
	__extract__(rows, cols)
		Raises a ValueError if not self.__has_sampled__. Otherwise returns geno[rows, ]
		and bv, ev, pheno sliced by [rows, cols]
	'''
	
	def __init__(self, M, N, mafs, betas, hsqs, e_var=None):	
		'''
		Initializes GWAS object and draws samples. The bulk of the simulation work happens,
		here, so a call to __init__ is likely to take some time.		
		'''
		self.M = M
		self.N = N
		self.mafs = mafs
		self.betas = np.matrix(betas-np.mean(betas))
		self.hsqs = np.matrix(hsqs)
		# normalize beta so that np.var(self.beta) = self.hsq
		self.betas = np.asarray(np.sqrt(hsqs))*np.asarray(self.betas)/ np.sqrt(np.var(self.betas)*self.M)	
		if self.hsqs.shape == (1,1): 
			self.num_phenos = 1
			self.e_var = 1 - hsqs
			self.betas = np.matrix(self.betas).reshape((self.M, 1))
		else:
			self.num_phenos = self.hsqs.shape[1]
			if e_var is None:
				raise ValueError('If num_phenos > 1, must specify e_var')

			self.e_var = np.matrix(e_var)
			# floating point bullshit
			if not np.all(np.diag(e_var) - np.squeeze(np.asarray(1-hsqs)) < 10**-6):
				raise ValueError('Environmental variance must equal 1 - hsq')
				
		self.__has_sampled__ = False
		NPH = self.N * self.num_phenos
		self.pheno = np.matrix(np.zeros(NPH)).reshape((N, self.num_phenos))
		self.geno = np.matrix(np.zeros((N,M)))
		self.bv = np.matrix(np.zeros(NPH)).reshape((N, self.num_phenos))
		self.ev = np.matrix(np.zeros(NPH)).reshape((N, self.num_phenos))
		self.__current_N__ = 0
		
	def sample(self):
		if not self.__has_sampled__:
			(geno, bv, ev, pheno) = self.__sample__(self.N-self.__current_N__)
		else:
			raise ValueError('All samples have been drawn')
		self.__append__(geno, bv, ev, pheno)
		
	def __sample__(self, N):
		geno = self.__draw_geno__(N)
		bv = self.__bv__(geno)
		ev = self.__ev__(N)
		pheno = self.__calc_pheno__(bv, ev)
		return (geno, bv, ev, pheno)

	def __draw_geno__(self, num_samples):
		genotypes = np.array([(np.random.binomial(2,p,num_samples) - 2*p)/np.sqrt(2*p*(1-p)) 
			for p in self.mafs])
		genotypes.resize(num_samples, self.M)
		return(genotypes)
	
	def __append__(self, geno, bv, ev, pheno):
		num_append = bv.shape[0]
		if not bv.shape == ev.shape == pheno.shape:
			raise ValueError('Inconsistent dimensions among bv/ev/pheno to be inserted')
		if not geno.shape[1] == self.M:
			raise ValueError('Genotypes to be inserted have incorrect number of SNPs')
		if not geno.shape[0] == num_append:
			raise ValueError('Inconsistent number of individuals to be inserted')
		max_N = self.N - self.__current_N__
		if num_append > max_N:
			raise ValueError('Cannot append more than {N} individuals'.format(N=max_N))
			
		start = self.__current_N__
		end = start + num_append
		self.geno[start:end,:] = geno
		self.bv[start:end,:] = bv
		self.ev[start:end,:] = ev
		self.pheno[start:end,:] = pheno
		self.__current_N__ += num_append
		if self.__current_N__ >= self.N:
			self.__has_sampled__ = True
		
	def __bv__(self, geno):
		return np.matrix(np.dot(geno, self.betas))
	
	def __ev__(self, num):
		mean = np.zeros(self.num_phenos)
		return np.matrix(sim.rnorm(num, mean, self.e_var)).reshape((num, self.num_phenos))
	
	def __insert__(self, (data)):
		(geno, bv, ev) = data
		if self.__has_sampled__:
			raise ValueError('Cannnot append after sampling')
		pheno = self.__calc_pheno__(bv, ev)
		self.__append__(geno, bv, ev, pheno)
		self.N_overlap = self.__current_N__
	
	def __extract__(self, rows, cols):
		# TODO: check more stuff
		if not self.__has_sampled__:
			raise ValueError('Must sample before calling __extract__()')
		
		rows = np.atleast_1d(np.array(rows)) # ugly ugly ugly
		cols = np.atleast_1d(np.array(cols)) # ugly ugly ugly
		geno = self.geno[rows,:]
		bv = self.bv[rows,:]
		bv = bv[:,cols]
		ev = self.ev[rows,:][:,cols]
		pheno = self.pheno[rows,:][:,cols]
		return (geno, bv, ev)
	
	def __sumstats__(self, which=None):
		if not self.__has_sampled__:
			self.sample()
			
		if not hasattr(self, '__betahat__') or not hasattr(self, '__chisq__'):
			pheno = np.squeeze(np.asarray(self.pheno[:,0]))
			norm_pheno = (pheno - np.mean(pheno)) / np.std(pheno)
			self.__betahat__ = np.dot(self.geno.T, norm_pheno).T / self.N
			self.__chisq__ = self.N*np.square(self.__betahat__)
			
		if which == 'chisq':
			return self.__chisq__
		elif which == 'betahat':
			return self.__betahat__
		else:
			raise ValueError('Which must be either chisq or betahat')
	
	@property
	def chisq(self):
		return self.__sumstats__(which='chisq')
	
	@property
	def betahat(self):
		return self.__sumstats__(which='betahat')
	
	def __calc_pheno__(self):
		raise NotImplementedError


class QT_GWAS(GWAS):
	'''
	Inherits from class GWAS. Quantitative trait GWAS with normally distributed 
	environmental effects and no ascertainment. 	
	
	Parameters and Attributes are the same as the parent GWAS class.
	
	Methods
	-------
	
	__calc_pheno__(bv,ev)  
		Overrides parent classCalculate phenotype as sum of breeding value and environmental 
		factors.
	
	'''

	def __calc_pheno__(self, bv, ev):
		'''
		Calculates phenotype as sum of breeding value and environmental factors.
		'''
		return bv + ev
	

class CC_GWAS(GWAS):
	'''
	Simulation class for ascertained case/control phenotypes, possibly with ascertainment
	depending on several phenotypes.
	
	Only methods and attributes that differ from the parent class are listed.

	Parameters
	----------
	
	M : int
		Number of bi-allelic SNPs sampled.
	N_cas: int
		Number of diploid cases sampled
	N_con : int
		Number of diploid controls sampled
	mafs : np.ndarray of floats with shape (M, ).
		Minor allele frequencies (must be in the interval (0,0.5]
	betas : np.ndarray of floats with shape (M, # phenotypes)
		Additive genetic effect sizes, possibly for several phenotypes if several phenotypes
		are needed for ascertainment.
	hsqs : float or np.ndarray of floats with shape (# phenotypes, )
		Heritability of liability all phenotypes needed for ascertainment. All values must lie
		in the interval [0,1].
	e_var : np.ndarray of floats with shape (# phenotypes, # phenotypes), optional
		Environmental covariance matrix (on liability scale). Required if # of phenotypes > 1.
		If # phenotypes == 1, then the environmental variance is set to 1 - hsqs.
	P : float or ndarray of floats with shape (# phenotypes, )
		Population prevalences of each phenotype
	chunk_size: int, default = 4000
		Number of samples to draw in each round of ascertainment. Does not affect the outcome
		of the simulation, but chunk_size > 1 does increase the speed of the simulation 
		significantly.

	Attributes
	----------
	
	geno : ndarray of floats with shape (N,M)
		Matrix of normalized genotypes. Accessed with self.__append__
	pheno : ndarray of floats with shape (N, # phenotypes) or (N, ) if # of phenotypes == 1
		Phenotypes on the observed (binary) scale. The phenotype used in association tests is 
		in the first column.	
	bv : ndarray of floats with shape (N, # phenotypes) or (N, ) if # of phenotypes == 1
		Breeding values on the liability scale.
	ev : ndarray of floats with shape (N, # phenotypes) or (N, ) if # of phenotypes == 1
		Environmental effects on the liability scale
	num_phenos : int
		Number of phenotypes. The first phenotype is used for association testing; the rest
		are used for ascertainment.
	thresh : float or ndarray of floats with shape (# phenotypes, )
		Liability thresholds
	N : int
		N_cas + N_con
	K : float
		Prevalence of association phenotype in-sample (N_cas / N)
	__z__ : float or ndarray of floats with shape (# phenotypes, )
		Height of standard normal at liability thresholds
	__current_N_cas__ : int
		Number of cases that have been sampled, either by calls to __sample__ or __insert__.
		Set equal to zero by __init__(), then modified by each subsequent call to __append__.
	__current_N_con__ : int
		Number of cases that have been sampled, either by calls to __sample__ or __insert__.
		Set equal to zero by __init__(), then modified by each subsequent call to __append__.
	__has_sampled__ : bool
		True if all samples have been collected has been called; otherwise false.	Set to False
		by __init__() then modified by __append__().
	-------
	
	sample()
		Draws (geno, bv, ev, pheno), where geno = an ndarray of genotypes with shape (N, M);
		bv = an ndarray of breeding values with shape (N, # phenotypes); ev = an ndarray 
		of environmental effects with shape (N, # phenotypes); pheno = an ndarray of shape 
		(N, # phenotypes). Calls __append__(geno, bv, ev, pheno)
	__extract__(num_con, cols)
		Raises a ValueError if self.__has_sampled__ is False or if num_con exceeds self.N_con.
		Otherwise selects num_con controls at random and returns genotypes plus columns cols 
		of bv, ev, pheno for these controls. Extends super.__extract__.
	__init__()
		Extends super.__init__() to deal with the extra parameters.
	__append__()
		Extends super.__append__() to tally number of appended cases and controls separately.
		Raises a ValueError if the number of individuals to append exceeds the total number 
		of individuals left to sample, but if the number of individuals to append is less than
		the total number of individuals left to sample but either the number of cases 
		or number of controls to append exceeds the number of cases or number of controls left
		to sample, respectively, then this function will append cases and controls in order
		until the limit is reached.
	__ascertainment_met__()
		Returns True if the required number of cases and controls have been sampled, False
		otherwise.
	__ascertain_cases__(draw)
		Assumes draw = (geno, bv, ev, pheno). Returns indices of those individuals for which 
		pheno[,0] == 1. Subclass and override for more complicated case ascertainment.
	__ascertain_controls__(draw)
		Assumes draw = (geno, bv, ev, pheno). Returns those individuals for which all 
		phenotypes are zero (healthy controls). Subclass and override for more complicated 
		control ascertainment.
	__calc_pheno__(bv, ev)
		Returns an ndarray of floats with shape (bv.shape[0], # phenotypes) or (bv.shape[0],)
		if # phenotypes == 1. Compares each liability with the corresponding liability 
		threshold in order to assign case/control status
	
	'''
	def __init__(self, M, N_cas, N_con, mafs, betas, hsqs, P, e_var=None, chunk_size=4000):
		self.M = M
		self.N_cas = N_cas
		self.N_con = N_con
		self.P = P
		self.chunk_size = chunk_size
		super(CC_GWAS, self).__init__(M, N_cas+N_con, mafs, betas, hsqs, e_var)
		self.K = N_cas / self.N
		# liability thresholds
		self.thresh = np.matrix(spstats.norm.isf(P)).reshape((1,self.num_phenos))
		# height of normal pdf at thresh
		self.__z__ = spstats.norm.pdf(self.thresh[0]) 
		self.__current_N_cas__ = 0
		self.__current_N_con__ = 0
	
	def __append__(self, geno, bv, ev, pheno):
		'''
		NOTE: controls are considered to be those individuals for whom assoc_pheno == 0
		
		If individuals to be appended are produced from __sample__ and ascertained as 
		controls via __ascertain_controls__, then individuals with assoc_pheno == 0 will
		be sent to append only if all other phenos are also zero.
		
		If individuals to be appended are produced via insert, then this may not hold. 
		'''
	
		# remove extra cases and controls from the list of samples to append
		remain_cas = self.N_cas - self.__current_N_cas__
		remain_con = self.N_con - self.__current_N_con__
		assoc_pheno = pheno[:,0]
		# cases
		ii_cas = assoc_pheno == 1
		jj = np.nonzero(ii_cas)[0]
		ii_cas[jj[:,remain_cas:len(ii_cas)]] = False
		# controls
		ii_con = assoc_pheno == 0
		jj = np.nonzero(ii_con)[0]
		ii_con[jj[:,remain_con:len(ii_con)]] = False
		ii = np.squeeze(np.asarray(np.logical_or(ii_cas, ii_con)))
		geno = geno[ii,:]; bv = bv[ii,:];	ev = ev[ii,:];	pheno = pheno[ii,:]
		super(CC_GWAS, self).__append__(geno, bv, ev, pheno)
		self.__current_N_cas__ += np.sum(ii_cas)
		self.__current_N_con__ += np.sum(ii_con)

	def sample(self):
		while not self.__ascertainment_met__():
			draw = super(CC_GWAS, self).__sample__(self.chunk_size)
			cas = self.__ascertain_cases__(draw)
			con = self.__ascertain_controls__(draw)
			asc = np.logical_or(cas, con) # ascertained samples
			if np.any(asc):
				self.__append__( *[x[asc,:] for x in draw] )
					
	def __ascertainment_met__(self):
		# have sampled exactly the desired number of cases and controls
		if self.__current_N_cas__ == self.N_cas and self.__current_N_con__ == self.N_con: 
			return True
		
		# next two for debugging
		elif self.__current_N_cas__ > self.N_cas or self.__current_N_cas__ < 0:
			error_msg = 'Found a bug: currentl {N} cases; self.N_cas is {P}'
			raise ValueError(error_msg.format(N=self.__current_N_cas__, P=self.N_cas))
		elif self.__current_N_con__ > self.N_con or self.__current_N_con__ < 0:
			error_msg = 'Found a bug: currentl {N} controls; self.N_con is {P}'
			raise ValueError(error_msg.format(N=self.__current_N_con__, P=self.N_con))
		
		# still need to sample more cases or controls
		else:
			return False
	
	def __ascertain_cases__(self, draw):
		pheno = draw[3]
		assoc_pheno = pheno[:,0]
		ii = assoc_pheno == 1
		return np.squeeze(np.asarray(ii))
		
	def __ascertain_controls__(self, draw):
		pheno = draw[3]
		ii = np.prod(pheno, axis=1) == 0
		return np.squeeze(np.asarray(ii))
		
	def __calc_pheno__(self, bv, ev):
		return (bv + ev > self.thresh).astype('float64')


class DoubleGWAS(object):
	'''
	(Minimal) parent class for all two-GWAS simulation classes	
	
	Methods
	-------
	
	__init__() needs to be implemented by subclasses
	sample():
		Draws samples for both constituent GWAS (with overlap).		
	betahat
		Returns ndarray with shape (M,2) consisting of slopes from marginal regression of 
		phenotype on genotype for all M SNPs.
	chisq
		Returns ndarray with shape (M,2) consisting of Armitage Trend Test statistics (equal
		to N * betahat ^ 2 from marginal regression of phenotype on genotype for all M SNPs.
	__sumstats__()
		Calculates and caches betahat or chisq if not already calculated. Returns 
		cached value of betahat or chisq
		
	'''
	
	def sample(self):
		rows = np.random.choice(np.arange(self.gwas1.N), replace=False, size=self.overlap)
		cols = 1
		self.__sample__(rows, cols)

	def __sample__(self, rows, cols): 
		self.gwas1.sample()
		self.gwas2.__insert__(self.gwas1.__extract__(rows, cols))
		self.gwas2.sample()
		
	def __sumstats__(self, which=None):
		if not hasattr(self, '__betahat__') or not hasattr(self, '__chisq__'):			
			# shape is # phenos, M
			self.__betahat__ = np.hstack((self.gwas1.betahat, self.gwas2.betahat))
			self.__chisq__ = np.square(self.__betahat__)
			self.__chisq__[:,0] *= self.gwas1.N
			self.__chisq__[:,1] *= self.gwas2.N
			
		if which == 'chisq':
			return self.__chisq__
		elif which == 'betahat':
			return self.__betahat__
		else:
			raise ValueError('Which must be either chisq or betahat')
	
	@property
	def chisq(self):
		return self.__sumstats__(which='chisq')
	
	@property
	def betahat(self):
		return self.__sumstats__(which='betahat')
	
	@property
	def betadot(self):
		return np.prod(self.__sumstats__(which="betahat"), axis=1)

	@property
	def chisqdot(self):
		return np.prod(self.__sumstats__(which="chisq"), axis=1)
		
	@property
	def chisq1(self):
		return self.__sumstats__(which="chisq")[:,0]
		
	@property
	def chisq2(self):
		return self.__sumstats__(which="chisq")[:,1]

	@property
	def betahat1(self):
		return self.__sumstats__(which="betahat")[:,0]
		
	@property
	def betahat2(self):
		return self.__sumstats__(which="betahat")[:,1]


class QT_QT_GWAS(DoubleGWAS):
	'''
	Simulation class with two QT simulations with no ascertainment and u.a.r. sample overlap
	
	Parameters
	----------
	
	M : int
		Number of bi-allelic SNPs sampled.
	N1,N2 : int
		Number of diploid individuals sampled.
	mafs, mafs2 : np.ndarray of floats with shape (M, ).
		Minor allele frequencies (must be in the interval (0,0.5]
	betas1, betas2 : np.ndarray of floats with shape (M, )
		Additive genetic effect sizes, only for the phenotypes in question, since there is 
		no ascertainment in this class
	hsq1, hsq2 : floats
		Heritability, must lie in the interval [0,1].
	e_var : np.ndarray of floats with shape (2,2)
		Environmental covariance matrix. 
	overlap : int, default = 0
		Number of overlapping individuals (chosen at random). Must be < min(N1, N2)
	
	Attributes
	----------
		
		gwas1, gwas2 : QT_GWAS
	
	'''
	def __init__(self, M, N1, N2, mafs, beta1, beta2, hsq1, hsq2, e_var, overlap):
		betas = np.matrix(np.vstack((beta1, beta2)).T)
		hsqs = np.matrix((hsq1, hsq2))
		self.overlap = overlap
		self.gwas1 = QT_GWAS(M, N1, mafs, betas, hsqs, e_var)
		self.gwas2 = QT_GWAS(M, N2, mafs, beta2, hsq2, float(e_var[1,1]))
		

class QT_CC_GWAS(DoubleGWAS):
	'''
	Simulation class with one QT simulation and one CC simulation. The CC simulation is not
	ascertained w.r.t. the association phenotype from the QT simulation. Any sample overlap 
	comes from selecting individuals from the QT simulation u.a.r. 
	
	Parameters
	----------
	
	M : int
		Number of bi-allelic SNPs sampled.
	N1 : 
		Number of diploid individuals sampled in QT GWAS.
	N_cas, N_con:
		Number of cases and controls sampled in CC GWAS.
	mafs, mafs2 : np.ndarray of floats with shape (M, ).
		Minor allele frequencies (must be in the interval (0,0.5]
	betas1, betas2 : np.ndarray of floats with shape (M, )..
		Additive genetic effect sizes, only for the phenotypes in question, since there is 
		no ascertainment in this class.
	hsq1, hsq2 : floats
		Heritability, must lie in the interval [0,1].
	e_var : np.ndarray of floats with shape (2,2)
		Environmental covariance matrix.
	P : float or ndarray of floats with shape (# phenotypes, )
		Population prevalences of each phenotype.
	overlap : int, default = 0
		Number of overlapping individuals (chosen at random). Must be < min(N1, N2)
	chunk_size : int, optional.
		
	Attriubtes
	----------
	
		gwas1 : QT_GWAS
		gwas2 : CC_GWAS
	
	Note that the second phenotype in gwas1 is assumed to be the association phenotype
	for gwas2.
	
	Methods
	-------
	
	sample() 
		Overrides DoubleGWAS.sample() so as to only include controls in the overlap set.
	
	'''
	def __init__(self, M, N1, N_cas, N_con, mafs, beta1, beta2, hsq1, hsq2, P, e_var,
		overlap, chunk_size=4000):
		betas = np.matrix(np.vstack((beta1, beta2)).T)
		hsqs = np.matrix((hsq1, hsq2))
		self.overlap = overlap
		self.gwas1 = QT_GWAS(M, N1, mafs, betas, hsqs, e_var)
		self.gwas2 = CC_GWAS(M, N_cas, N_con, mafs, beta2, hsq2, P, float(e_var[1,1]), 
			chunk_size=chunk_size)
		
		
class CC_CC_GWAS(DoubleGWAS):
	'''
	Simulation class with two CC simulations. 
	
	The functionality implemented here is limited to the 'healthy controls with sample 
	overlap' model of a disease genetics GWAS consortium (which could perhaps be described
	more succinctly as the PGC model). The first phenotype is the association phenotype
	for the first GWAS; the second phenotype is the association phenotype for the second
	GWAS; all subsequent phenotypes are not used for association statistics but are used for
	ascertainment. Both GWAS are ascertained for both association phenotypes
	
	Parameters
	----------
	
	M : int
		Number of bi-allelic SNPs sampled.
	N_cas : np.ndarray with shape(2,)
		Number of cases sampled for each GWAS
	N_con : np.ndarray with shape (2,)
		Number of controls sampled for each GWAS
	mafs : np.ndarray of floats with shape (M, 1).
		Minor allele frequencies (must be in the interval (0,0.5]
	betas : np.matrix of floats with shape (M, # phenos)
		Additive genetic effect sizes for all phenotypes. First column is effects for assoc
		phenotype 1; second column is effects for assoc phenotype 2; all other columns
		are for ascertainment phenotypes
	hsqs : np.ndarray of floats with shape (# phenos, )
		Heritability, must lie in the interval [0,1].
	e_var : symmetric, positive-definite np.matrix with shape (# phenos, # phenos)
		Environmental covariance matrix.
	P : np.ndarray of floats with shape (# phenotypes, )
		Population prevalences of each phenotype
	overlap : int, default = 0 
		Number of overlapping controls (chosen at random). Must be < min(N_con)
	chunk_size : int, optional
		
	Attributes
	----------
		
	gwas1 : CC_GWAS
	gwas2 : CC_GWAS
		
	Methods
	-------
	
	sample()
		Extends DoubleGWAS.sample.
	
	'''
	def __init__(self,M,N_cas,N_con,mafs,betas,hsqs,P,e_var,overlap,chunk_size=4000):
		self.overlap = overlap
		hsqs = np.matrix(hsqs)
		cs = chunk_size
		self.gwas1 = CC_GWAS(M,N_cas[0],N_con[0],mafs,betas,hsqs,P,e_var,chunk_size=cs)
		# TODO: this assumes that betas has at least 2 cols, which should be  checked 
		# switch col1 and col2 for gwas2
		betas[:,(0,1)] = betas[:,(1,0)]
		self.gwas2 = CC_GWAS(M,N_cas[1],N_con[1],mafs,betas,hsqs,P,e_var,chunk_size=cs)

	def sample(self):
		con_indices = np.nonzero(self.gwas1.pheno[:,0] == 0)[0]
		con_indices = np.squeeze(np.asarray(con_indices))
		rows = np.random.choice(con_indices, replace=False, size=self.overlap)
		# flip the order of phenotypes 1 and 2
		cols = np.hstack(( (1,0), np.arange(2, self.gwas1.num_phenos)))
		self.__sample__(rows, cols)