from __future__ import division
import numpy as np
import pandas as pd
from scipy.special import chdtri
import gzip


def filter_df(df, colname, pred):
	'''
	Filters df down to those rows where pred applied to colname returns True
	
	Parameters
	----------
	df : pd.DataFrame
		Data frame to filter.
	colname : string
		Name of a column in df.
	pred : function
		Function that takes one argument (the data type of colname) and returns True or False.
		
	Returns
	-------
	df2 : pd.DataFrame
		Filtered version of df. 
	
	'''
	if colname in df.columns:
		df2 = df[pred( df[colname] )]
	else: 
		raise ValueError('Cannot find a column named {C}.'.format(C=colname))

	return df2


# input checking functions 

def check_dir(dir):
	'''
	Checks that directions of effect are sensible. Nonsense values should have been caught
	already by coercion to int.
	
	'''
	c1 = dir != 1
	c2 = dir != -1
	if np.any(np.logical_and(c1, c2)):	
		raise ValueError('DIR entry not equal to +/- 1.')


def check_rsid(rsids):
	'''
	Checks that rs numbers are sensible.
	
	'''
	# check for rsid = .
	#if np.any(rsids == '.'):
	#	raise ValueError('Some SNP identifiers are set to . (a dot).')
	
	# check for duplicate rsids
	#if np.any(rsids.duplicated('SNP')):
	#	raise ValueError('Duplicated SNP identifiers.')

	
def check_pvalue(P):
	'''
	Checks that P values are sensible. Nonsense values should have been caught already by 
	coercion to float.
	
	'''
	# check for missing 
	if np.any(np.isnan(P)):
		raise ValueError('Missing P-values')
	
	# check for P outside of the range (0,1]
	if np.max(P) > 1:
		raise ValueError('P-values cannot be > 1.')
	if np.min(P) <= 0:
		raise ValueError('P values cannot be <= 0')


def check_chisq(chisq):
	'''
	Checks that chi-square statistics are sensible. Nonsense values should have been caught
	already by coercion to float.
	
	'''
	if np.any(np.isnan(chisq)):
		raise ValueError('Missing chi-square statistics.')
	
	# check for chisq outside of the range [0,Inf)
	if np.max(chisq) == float('inf'):
		raise ValueError('Infinite chi-square statistics.')
	if np.min(chisq) < 0:
		raise ValueError('Negative chi-square statistics')


def check_maf(maf):
	'''
	Checks that MAFs are sensible. Nonsense values should have been caught already by 
	coercion to float.

	'''
	if np.any(np.isnan(maf)):
		raise ValueError('Missing values in MAF.')
	
	# check for MAF outside of the range (0,1)
	if np.max(maf) >= 1:
		raise ValueError('MAF >= 1.')
	if np.min(maf) <= 0:
		raise ValueError('MAF <= 0.')
	
	
def check_N(N):
	'''
	Checks that sample sizes are sensible. Nonsense values should have been caught already 
	by coercion to int.
	
	'''
	if np.min(N) < 0:
		raise ValueError('Negative N.')


# parsers

def chisq(fh):
	'''
	Parses .chisq files. See docs/file_formats_sumstats.txt
	
	'''
	dtype_dict = {
#		'CHR': str,
		'SNP': str,
#		'CM': float,
#		'BP': int,
		'P': float,
		'CHISQ': float,
		'N': float,
		'MAF': float,
		'INFO': float,
	}
	if fh.endswith('gz'):
		openfunc = gzip.open
		compression='gzip'
	else:
		openfunc = open
		compression=None
		
	colnames = openfunc(fh,'rb').readline().split()
	usecols = ['SNP','P','CHISQ','N','MAF','INFO']	
	usecols = [x for x in usecols if x in colnames]
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict, compression=compression)
	except AttributeError as e:
		raise AttributeError('Improperly formatted chisq file: '+str(e.args))

	try:
		check_N(x['N'])	
	except KeyError as e:
		raise KeyError('No column named N in .betaprod: '+str(e.args))

	x['N'] = x['N'].astype(float)
	
	try:
		check_rsid(x['SNP'])
	except KeyError as e:
		raise KeyError('No column named SNP in .betaprod: '+str(e.args))
	
	if 'MAF' in x.columns:
		check_maf(x['MAF'])
		x['MAF'] = np.fmin(x['MAF'], 1-x['MAF'])
	
	if 'P' in x.columns:
		check_pvalue(x['P'])
		x['P'] = chdtri(1, x['P']); 
		x.rename(columns={'P': 'CHISQ'}, inplace=True)
	elif 'CHISQ' in x.columns:
		check_chisq(x['CHISQ'])
	else:
		raise ValueError('.chisq file must have a column labeled either P or CHISQ.')

	return x
	
def betaprod_fromchisq(chisq1, chisq2, allele1, allele2):
	''' 
	Makes a betaprod data frame from two chisq files and two trait-increasing-allele files.
	The allele files have a SNP column and an INC_ALLELE, and INC_ALLELE.
	'''

	df_chisq1 = pd.read_csv(chisq1, delim_whitespace=True)
	df_chisq1['BETAHAT1'] = np.sqrt(df_chisq1['CHISQ'])/np.sqrt(df_chisq1['N'])
	df_chisq1.rename(columns={'N':'N1'},inplace=True)
	del df_chisq1['CHISQ']

	df_chisq2 = pd.read_csv(chisq2, delim_whitespace=True)
	df_chisq2['BETAHAT2'] = np.sqrt(df_chisq2['CHISQ'])/np.sqrt(df_chisq2['N'])
	df_chisq2.rename(columns={'N':'N2'},inplace=True)
	del df_chisq2['CHISQ']
	df_merged = pd.merge(df_chisq1,df_chisq2, how='inner', on='SNP')
	
	df_allele1 = pd.read_csv(allele1, delim_whitespace=True)
	df_allele1.rename(columns={'INC_ALLELE':'INC_ALLELE1'},inplace=True)
	df_merged = pd.merge(df_merged,df_allele1,how='inner', on='SNP')

	df_allele2 = pd.read_csv(allele2, delim_whitespace=True)
	df_allele2.rename(columns={'INC_ALLELE':'INC_ALLELE2'},inplace=True)
	df_merged = pd.merge(df_merged,df_allele2,how='inner', on='SNP')

	df_merged['BETAHAT2'] *= (-1)**(df_merged.INC_ALLELE1 != df_merged.INC_ALLELE2)

	return df_merged


def betaprod(fh):
	'''
	Parses .betaprod files. See docs/file_formats_sumstats.txt
	
	'''
	dtype_dict = {
#		'CHR': str,
		'SNP': str,
#		'CM': float,
#		'BP': int,
		'P1': float,
		'CHISQ1': float,
		'DIR1': float,
		'N1': int, # cast to int for typechecking, then switch to float later for division
		'P2': float,
		'CHISQ2': float,
		'DIR2': float,
		'N2': int,
		'INFO1': float,
		'INFO2': float,
		'MAF1': float,
		'MAF2': float
	}
	if fh.endswith('gz'):
		openfunc = gzip.open
		compression='gzip'
	else:
		openfunc = open
		compression=None

	colnames = openfunc(fh,'rb').readline().split()
	usecols = [x+str(i) for i in xrange(1,3) for x in ['DIR','P','CHISQ','N','MAF','INFO']]
	usecols.append('SNP')
	usecols = [x for x in usecols if x in colnames]
	try:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
			dtype=dtype_dict, compression=compression)
	except AttributeError as e:
		raise AttributeError('Improperly formatted betaprod file: '+str(e.args))
		
	try:
		check_rsid(x['SNP'])
	except KeyError as e:
		raise KeyError('No column named SNP in .betaprod: '+str(e.args))
	
	for i in ['1','2']:
		N='N'+i; P='P'+i; CHISQ='CHISQ'+i; DIR='DIR'+i; MAF='MAF'+i; INFO='INFO'+i
		BETAHAT='BETAHAT'+i
		try:
			check_N(x[N])
		except KeyError as e:
			raise KeyError('No column named {N} in .betaprod: '.format(N=N)+str(e.args))
		x[N] = x[N].astype(float)	
		try:
			check_dir(x[DIR])
		except KeyError as e:
			raise KeyError('No column named {D} in .betaprod: '.format(D=DIR)+str(e.args))
	
		if CHISQ in x.columns:
			check_chisq(x[CHISQ])
			betahat = np.sqrt(x[CHISQ]/x[N]) * x[DIR]
			x[CHISQ] = betahat
			x.rename(columns={CHISQ: BETAHAT}, inplace=True)
		elif P in x.columns:
			check_pvalue(x[P])
			betahat = np.sqrt(chdtri(1, x[P])/x[N])	* x[DIR]
			x[P] = betahat
			x.rename(columns={P: BETAHAT}, inplace=True)
		else:
			raise ValueError('No column named P{i} or CHISQ{i} in betaprod.'.format(i=i))

		del x[DIR]
		if MAF in x.columns:
			check_maf(x[MAF])
			x[MAF]  = np.minimum(x[MAF], 1-x[MAF])
		
	return x

	
def ldscore(fh, num=None):
	'''
	Parses .l2.ldscore files. See docs/file_formats_ld.txt

	If num is not None, parses .l2.ldscore files split across [num] chromosomes (e.g., the 
	output of parallelizing ldsc.py --l2 across chromosomes).

	'''
						
	parsefunc = lambda y, compression : pd.read_csv(y, header=0, delim_whitespace=True,
		compression=compression).drop(['CHR','BP','CM','MAF'], axis=1)
	
	if num is not None:
		try:
			suffix = '.l2.ldscore.gz'
			if '@' in fh:
				full_fh = fh.replace('@', '1') + suffix
			else:
				full_fh = fh + '1' + suffix
	
			x = open(full_fh, 'rb')
			x.close()
			compression = 'gzip'
		except IOError:
			suffix = '.l2.ldscore'

			if '@' in fh:
				full_fh = fh.replace('@', '1') + suffix
			else:
				full_fh = fh + '1' + suffix
			x = open(full_fh, 'rb')
			x.close()
			compression = None			
		
		if '@' in fh:
			chr_ld = [parsefunc(fh.replace('@',str(i))+suffix, compression) for i in xrange(1,num+1)]
		else:
			chr_ld = [parsefunc(fh + str(i) + suffix, compression) for i in xrange(1,num+1)]

		x = pd.concat(chr_ld)
	
	else:
		try:
			full_fh = fh + '.l2.ldscore.gz'
			open(full_fh, 'rb')
			compression = 'gzip'
		except IOError:
			full_fh = fh + '.l2.ldscore'
			open(full_fh, 'rb')
			compression = None			
		
		x = parsefunc(full_fh, compression)
	
	ii = x['SNP'] != '.'
	x = x[ii]	
	check_rsid(x['SNP']) 
	for col in x.columns[1:]:
		x[col] = x[col].astype(float)
	
	return x
	
	
def l1(fh, num=None):
	'''
	Parses .l1.ldscore files. See docs/file_formats_ld.txt

	If num is not None, parses .l1.ldscore files split across [num] chromosomes (e.g., the 
	output of parallelizing ldsc.py --l1 across chromosomes).

	'''
						
	parsefunc = lambda y, compression : pd.read_csv(y, header=0, delim_whitespace=True,
		compression=compression).drop(['CHR','BP','CM','MAF'], axis=1)
	
	if num is not None:
		try:
			suffix = '.l1.ldscore.gz'
			if '@' in fh:
				full_fh = fh.replace('@', '1') + suffix
			else:
				full_fh = fh + '1' + suffix
	
			x = open(full_fh, 'rb')
			x.close()
			compression = 'gzip'
		except IOError:
			suffix = '.l1.ldscore'

			if '@' in fh:
				full_fh = fh.replace('@', '1') + suffix
			else:
				full_fh = fh + '1' + suffix
			x = open(full_fh, 'rb')
			x.close()
			compression = None			
		
		if '@' in fh:
			chr_ld = [parsefunc(fh.replace('@',str(i))+suffix, compression) for i in xrange(1,num+1)]
		else:
			chr_ld = [parsefunc(fh + str(i) + suffix, compression) for i in xrange(1,num+1)]

		x = pd.concat(chr_ld)
	
	else:
		try:
			full_fh = fh + '.l1.ldscore.gz'
			open(full_fh, 'rb')
			compression = 'gzip'
		except IOError:
			full_fh = fh + '.l1.ldscore'
			open(full_fh, 'rb')
			compression = None			
		
		x = parsefunc(full_fh, compression)
	
	ii = x['SNP'] != '.'
	x = x[ii]	
	check_rsid(x['SNP']) 
	for col in x.columns[1:]:
		x[col] = x[col].astype(float)
	
	return x


def M(fh, num=None, N=2):
	'''
	Parses .l{N}.M files. See docs/file_formats_ld.txt.
	
	If num is not none, parses .l2.M files split across [num] chromosomes (e.g., the output 
	of parallelizing ldsc.py --l2 across chromosomes).

	'''
	parsefunc = lambda y : [float(z) for z in open(y, 'r').readline().split()]
	suffix = '.l'+str(N)+'.M'
	if num is not None:
		if '@' in fh:
			x = np.sum([parsefunc(fh.replace('@',str(i))+suffix) for i in xrange(1,num+1)], axis=0)
		else:
			x = np.sum([parsefunc(fh+str(i)+suffix) for i in xrange(1,num+1)], axis=0)
	else:
		x = parsefunc(fh + suffix)
		
	return x
	

def __ID_List_Factory__(colnames, keepcol, fname_end, header=None, usecols=None):
	
	class IDContainer(object):
		
		def __init__(self, fname):
			self.__usecols__ = usecols
			self.__colnames__ = colnames
			self.__keepcol__ = keepcol
			self.__fname_end__ = fname_end
			self.__header__ = header
			self.__read__(fname)

			if self.__colnames__:
				if 'SNP' in self.__colnames__:
					check_rsid(self.df['SNP'])
				
			self.n = len(self.IDList)

		def __read__(self, fname):
			end = self.__fname_end__
			if end and not fname.endswith(end):
				raise ValueError('{f} filename must end in {f}'.format(f=end))
			
			if fname.endswith('gz'):
				self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__, 
					delim_whitespace=True, compression='gzip')
			elif fname.endswith('bz2'):
				self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__, 
					delim_whitespace=True, compression='bz2')
			else:
				self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__, 
					delim_whitespace=True)
			
			
			#if np.any(self.df.duplicated(self.df.columns[self.__keepcol__])):
			#	raise ValueError('Duplicate Entries in Filter File')

			if self.__colnames__: 
				self.df.columns = self.__colnames__

			self.IDList = self.df.iloc[:,[self.__keepcol__]].astype('object')
		
		def loj(self, externalDf):
			'''
			Returns indices of those elements of self.IDList that appear in exernalDf
			'''
			r = externalDf.columns[0]
			l = self.IDList.columns[0]
			merge_df = externalDf.iloc[:,[0]]
			merge_df['keep'] = True
			z = pd.merge(self.IDList, merge_df, how='left',left_on=l, right_on=r, 
				sort=False)
			ii = z['keep'] == True	
			return np.nonzero(ii)[0]

	return IDContainer


PlinkBIMFile = __ID_List_Factory__(['CHR', 'SNP','CM','BP','A1','A2'],1,'.bim',usecols=[0,1,2,3,4,5])
VcfSNPFile = __ID_List_Factory__(['CHR','BP','SNP','CM'],2,'.snp',usecols=[0,1,2,3])
PlinkFAMFile = __ID_List_Factory__(['IID'],0,'.fam',usecols=[1])
VcfINDFile = __ID_List_Factory__(['IID'],0,'.ind',usecols=[0])
FilterFile = __ID_List_Factory__(['ID'],0,None,usecols=[0])
AnnotFile = __ID_List_Factory__(None,2,'.annot',header=0,usecols=None)
