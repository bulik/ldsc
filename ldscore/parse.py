'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module contains functions for parsing various ldsc-defined file formats

'''

from __future__ import division
import numpy as np
import pandas as pd
from scipy.special import chdtri
import gzip, bz2
import os


def get_compression(fh):
	'''Which sort of compression should we use with pd.read_csv?'''
	if fh.endswith('gz'):
		compression='gzip'
	elif fh.endswith('bz2'):
		compression='bz2'
	elif fh.endswith('pickle'):
		compression='pickle'
	else:
		compression=None
	return compression


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
	#		raise ValueError('Some SNP identifiers are set to . (a dot).')
	
	# check for duplicate rsids
	#if np.any(rsids.duplicated('SNP')):
	#		raise ValueError('Duplicated SNP identifiers.')

		
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


def cut_cts(vec, breaks):
	'''Cuts vectors for --cts-bin.'''
	
	max_cts = np.max(vec)
	min_cts = np.min(vec)
	cut_breaks = list(breaks)
	name_breaks = list(cut_breaks)
	if np.all(cut_breaks >= max_cts) or np.all(cut_breaks <= min_cts):
		raise ValueError('All breaks lie outside the range of the cts variable.')

	if np.all(cut_breaks <= max_cts):
		name_breaks.append(max_cts)
		cut_breaks.append(max_cts+1)
			
	if np.all(cut_breaks >= min_cts):		
		name_breaks.append(min_cts)
		cut_breaks.append(min_cts-1)

	name_breaks.sort()
	cut_breaks.sort()				
	n_breaks = len(cut_breaks)
	# so that col names are consistent across chromosomes with different max vals
	name_breaks[0] = 'min'
	name_breaks[-1] = 'max'
	name_breaks = [str(x) for x in name_breaks]
	labs = [name_breaks[i]+'_'+name_breaks[i+1] for i in xrange(n_breaks-1)]
	cut_vec = pd.Series(pd.cut(vec, bins=cut_breaks, labels=labs))
	return cut_vec


# parsers

def read_cts(fh, match_snps):
	'''Reads files for --cts-bin.'''
	comp = get_compression(fh)
	cts = pd.read_csv(fh, header=None, delim_whitespace=True, compression=comp)
	cts.rename( columns={0: 'SNP', 1: 'ANNOT'}, inplace=True)
	check_rsid(cts.SNP)
	snp = cts.SNP.values
	if len(snp) != len(match_snps) or np.any(snp != match_snps):
		msg = 'All --cts-bin files must contain the same SNPs in the same '
		msg += 'order as the .bim file.'
		raise ValueError(msg)

	return cts.ANNOT.values


def chisq(fh, require_alleles=False, keep_na=False, check=True):

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
		'INC_ALLELE': str,
		'DEC_ALLELE': str
	}
	
	compression = get_compression(fh)
	if compression != 'pickle':
		if compression == 'gzip':
			openfunc = gzip.open
		elif compression == 'bz2':
			raise NotImplementedError('bzipped chisq files not currently supported.')
		else:
			openfunc = open
		
		colnames = openfunc(fh,'rb').readline().split()
		if require_alleles:
			usecols = ['SNP','P','CHISQ','N','MAF','INFO','INC_ALLELE','DEC_ALLELE']	
		else:
			usecols = ['SNP','P','CHISQ','N','MAF','INFO']	
		
		usecols = [x for x in usecols if x in colnames]
		try:
			x = pd.read_csv(fh, header=0, delim_whitespace=True, usecols=usecols, 
				dtype=dtype_dict, compression=compression)
		except AttributeError as e:
			raise AttributeError('Improperly formatted chisq file: '+str(e.args))
			
	else: 
		x = pd.read_pickle(fh).reset_index(drop=True)
	
	if 'P' in x.columns:
		ii = x.P.notnull()
		check_pvalue(x.loc[ii,'P'])
		x.loc[ii,'P'] = chdtri(1, x.loc[ii,'P'])
		x.rename(columns={'P': 'CHISQ'}, inplace=True)
			
	if check:
		'''
		Check that all the columns make sense. These checks are somewhat slow, and are 
		completely redundant for chisq files generated using sumstats_to_chisq.py.
		'''
		try:
			check_N(x['N']) 
		except KeyError as e:
			raise KeyError('No column named N in .chisq: '+str(e.args))

		x['N'] = x['N'].astype(float)
	
		try:
			check_rsid(x.SNP)
		except KeyError as e:
			raise KeyError('No column named SNP in .chisq: '+str(e.args))
	
		ii = x.N.notnull()
		if 'MAF' in x.columns:
			maf = x.loc[ii, 'MAF']
			check_maf(maf)
			x.loc[ii,'MAF'] = np.fmin(maf, 1-maf)	
		elif 'CHISQ' in x.columns:
			check_chisq(x.loc[ii,'CHISQ'])
		else:
			raise ValueError('.chisq file must have a column labeled either P or CHISQ.')
		
		if require_alleles:
			'''
			Checks that only matter for rg estimation, where the alleles need to be aligned to
			the same ref allele. These checks are redundant for chisq files generated using
			sumstats_to_chisq.py and --merge-alleles.
					
			'''
			if not ('INC_ALLELE' in x.columns and 'DEC_ALLELE' in x.columns):
				msg = '.chisq file must have an INC_ALLELE and DEC_ALLELE column for rg estimation.'
				raise ValueError(msg)

			x.INC_ALLELE[ii] = x.INC_ALLELE[ii].apply(lambda y: y.upper())
			y = x.INC_ALLELE[ii].unique()
			p = ( (y == 'A') | (y == 'C') | (y == 'T') | (y == 'G')).all()
			if not p:
				raise ValueError('INC_ALLELE column must contain only A/C/T/G. Do you have indels?')

			x.DEC_ALLELE[ii] = x.DEC_ALLELE[ii].apply(lambda y: y.upper())
			y = x.DEC_ALLELE[ii].unique()
			p = ( (y == 'A') | (y == 'C') | (y == 'T') | (y == 'G')).all()
			if not p:
				raise ValueError('DEC_ALLELE column must contain only A/C/T/G. Do you have indels?')

			if np.any(x.INC_ALLELE[ii] == x.DEC_ALLELE[ii]):
				raise ValueError('INC_ALLELE cannot equal DEC_ALLELE.')
	
	if not keep_na:
		x = x[ii]

	return x


def ldscore_fromfile(fh, num=None):
	'''Sideways concatenation of a list of LD Scores from a file.'''
	f = open(fh,'r')
	lines = [x.rstrip('\n') for x in f.readlines()]
	return ldscore_fromlist(lines, num)


def annot_fromfile(fh, num=None, frqfile=None):
	'''Sideways concatenation of a list of annot matrices from a file.'''
	f = open(fh,'r')
	lines = [x.rstrip('\n') for x in f.readlines()]
	return annot(lines, num, frqfile)


def ldscore_fromlist(flist, num=None):
	'''Sideways concatenation of a list of LD Score files.'''
	ldscore_array = []
	for i,fh in enumerate(flist):						
		fh = fh.rstrip('\n')
		y = ldscore(fh, num)
		if i > 0:
			if not np.all(y.SNP == ldscore_array[0].SNP):
				msg = 'All files in --ref-ld-list or --ref-ld-file must contain the same SNPs '
				msg += 'in the same order.'
				raise ValueError(msg)
					
			y = y.ix[:,1:len(y.columns)] # remove SNP column
		
		# force unique column names
		new_col_dict = {c: c+'_'+str(i) for c in y.columns if c != 'SNP'}
		y.rename(columns=new_col_dict, inplace=True)
		ldscore_array.append(y) 
			
	x = pd.concat(ldscore_array, axis=1) 
	return x


def which_compression(fh):
	'''Given a file prefix, figure out what sort of compression to use'''
	if os.access(fh+'.pickle', 4):
		suffix = '.pickle'
		compression = 'pickle'
	elif os.access(fh+'.bz2', 4):
		suffix = '.bz2'
		compression = 'bz2'
	elif os.access(fh+'.gz', 4):
		suffix = '.gz'
		compression = 'gzip'
	elif os.access(fh, 4):
		compression = None
		suffix = ''
	else:
		msg = 'Could not open {F}[.pickle/gz/bz2]'
		raise IOError(msg.format(F=fh))
			
	return [suffix, compression]


def l2_parser(fh, compression):
	'''For parsing LD Score files'''
	if compression == "gzip" or compression == 'bz2' or compression == None:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, compression=compression)
	elif compression == 'pickle':
		x = pd.read_pickle(fh).reset_index(drop=True)
					
	return x.drop(['MAF','CM'], axis=1)


def annot_parser(fh, compression, frqfile_full=None, compression_frq=None):
	'''For parsing LD Score files'''
	if compression == "gzip" or compression == 'bz2' or compression == None:
		x = pd.read_csv(fh, header=0, delim_whitespace=True, compression=compression)
	elif compression == 'pickle':
		x = pd.read_pickle(fh)
	df_annot = x.drop(['CHR','BP','CM'], axis=1)

	for name in df_annot.columns[1:]:
		df_annot[name] = df_annot[name].astype(float)
	
	if frqfile_full is not None:
		df_frq = frq_parser(frqfile_full, compression_frq) 
		frq = np.array(df_frq['FRQ'])
		df_annot = df_annot[frq > 0.05]
		df_annot = df_annot[frq < 0.95]

	return df_annot


def frq_parser(fh, compression):
	'''For parsing frequency files'''
	if compression == "gzip" or compression == 'bz2' or compression == None:
		x = pd.read_csv(fh, header=0, delim_whitespace=True,		compression=compression)
	elif compression == 'pickle':
		x = pd.read_pickle(fh)
	return x
				
				
def ldscore(fh, num=None):
	'''
	Parses .l2.ldscore files. See docs/file_formats_ld.txt

	If num is not None, parses .l2.ldscore files split across [num] chromosomes (e.g., the 
	output of parallelizing ldsc.py --l2 across chromosomes).

	'''
											
	if num is not None:
		'''22 files, one for each chromosome'''
		suffix = '.l2.ldscore'
		if '@' in fh:
			first_fh = fh.replace('@', '1')+suffix
		else:
			first_fh = fh+'1'+suffix
		
		[s, compression] = which_compression(first_fh)
		suffix += s			
		if '@' in fh:
			chr_ld = [l2_parser(fh.replace('@',str(i))+suffix, compression) for i in xrange(1,num+1)]
		else:
			chr_ld = [l2_parser(fh + str(i) + suffix, compression) for i in xrange(1,num+1)]
		
		# automatically sorted by chromosome
		x = pd.concat(chr_ld)
 		x = x.drop(['CHR','BP'], axis=1)
	
	else:
		'''Just one file'''
		suffix = '.l2.ldscore'
		[s, compression] = which_compression(fh+suffix)
		suffix += s
		x = l2_parser(fh+suffix, compression)
 		# may need to sort
 		x = x.sort(['CHR','BP'])
 		x = x.drop(['CHR','BP'], axis=1)
 		
	x = x.drop_duplicates(subset='SNP')
	x = x[x.SNP != '.']			
	#check_rsid(x.index) 
	#for col in x.columns[1:]:
	#		 x[col] = x[col].astype(float)
	
	return x		
				
def annot(fh_list, num=None, frqfile=None):

	'''
	Parses .annot files and returns an overlap matrix. See docs/file_formats_ld.txt

	If num is not None, parses .annot files split across [num] chromosomes (e.g., the 
	output of parallelizing ldsc.py --l2 across chromosomes).

	'''
		
	if num is not None:
		'''22 files, one for each chromosome'''
		annot_suffix = ['.annot' for fh in fh_list]
		annot_compression = []
		for (i,fh) in enumerate(fh_list):
			if '@' in fh:
					first_fh = fh.replace('@', '1')+annot_suffix[i]
			else:
					first_fh = fh+'1'+annot_suffix[i]
			[annot_s,annot_comp_single] = which_compression(first_fh)
			annot_suffix[i] += annot_s 
			annot_compression.append(annot_comp_single)

		if frqfile is not None:
			frq_suffix = '.frq'
			if '@' in frqfile:
				first_frqfile = frqfile.replace('@', '1')+frq_suffix
			else:
				first_frqfile = frqfile+'1'+frq_suffix
			[frq_s, frq_compression] = which_compression(first_frqfile)
			frq_suffix += frq_s			
		
		y = []
		M_tot = 0
		for chr in xrange(1,num+1):

			if '@' in fh_list[0]:
				if frqfile is not None:
					df_annot_chr_list = [annot_parser(fh.replace('@',str(chr))+annot_suffix[i], annot_compression[i],
						frqfile.replace('@',str(chr))+frq_suffix,frq_compression) for (i,fh) in enumerate(fh_list)]
				else:
					df_annot_chr_list = [annot_parser(fh.replace('@',str(chr))+annot_suffix[i], annot_compression[i])
						for (i,fh) in enumerate(fh_list)]
			else:
				if frqfile is not None:
					df_annot_chr_list = [annot_parser(fh + str(chr) + annot_suffix[i], annot_compression[i], 
						frqfile+str(chr)+frq_suffix, frq_compression) for (i,fh) in enumerate(fh_list)]
				else:
					df_annot_chr_list = [annot_parser(fh + str(chr) + annot_suffix[i], annot_compression[i])
						for (i,fh) in enumerate(fh_list)]
			annot_matrix_chr_list = [np.matrix(df_annot_chr.ix[:,1:]) for df_annot_chr in df_annot_chr_list]
			annot_matrix_chr = np.hstack(annot_matrix_chr_list)
			y.append(np.dot(annot_matrix_chr.T, annot_matrix_chr))
			M_tot += len(df_annot_chr_list[0])
			del df_annot_chr
			del annot_matrix_chr

		x = sum(y)
	
	else:

		'''Just one file'''
		annot_suffix = ['.annot' for fh in fh_list]
		annot_compression = []
		for (i,fh) in enumerate(fh_list):
			[annot_s,annot_comp_single] = which_compression(fh+annot_suffix[i])
			annot_suffix[i] += annot_s 
			annot_compression.append(annot_comp_single)

		if frqfile is not None:
			frq_suffix = '.frq'
			[frq_s, frq_compression] = which_compression(frqfile+frq_suffix)
			frq_suffix += frq_s 

			df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i], 
										frqfile+frq_suffix, frq_compression) for (i,fh) in enumerate(fh_list)]
	
		else:
			df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i]) 
				for (i,fh) in enumerate(fh_list)]

		annot_matrix_list = [np.matrix(y.ix[:,1:]) for y in df_annot_list]
		annot_matrix = np.hstack(annot_matrix_list)

		x = np.dot(annot_matrix.T,annot_matrix)
		M_tot = len(df_annot_list[0])

	return [x, M_tot]
		
		
def M(fh, num=None, N=2, common=None):
	'''
	Parses .l{N}.M files. See docs/file_formats_ld.txt.

	If num is not none, parses .l2.M files split across [num] chromosomes (e.g., the output 
	of parallelizing ldsc.py --l2 across chromosomes).

	'''
	parsefunc = lambda y : [float(z) for z in open(y, 'r').readline().split()]
	if common:
		suffix = '.l'+str(N)+'.M_5_50'
	else:
		suffix = '.l'+str(N)+'.M'
	if num is not None:
		if '@' in fh:
			x = np.sum([parsefunc(fh.replace('@',str(i))+suffix) for i in xrange(1,num+1)], axis=0)
		else:
			x = np.sum([parsefunc(fh+str(i)+suffix) for i in xrange(1,num+1)], axis=0)
	else:
		x = parsefunc(fh + suffix)
	
	return x


def M_fromfile(flist, num=None, N=2, common = None):
	f = open(flist,'r')
	flist = [x.rstrip('\n') for x in f.readlines()]
	return M_fromlist		(flist, num, N, common)
		

def M_fromlist(flist, num=None, N=2, common = None):
	M_annot = np.hstack([M(fh.rstrip('\n'), num, N, common) for fh in flist])
	return M_annot


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
			
			comp = get_compression(fname)
			self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__, 
					delim_whitespace=True, compression=comp)
			#if np.any(self.df.duplicated(self.df.columns[self.__keepcol__])):
			#		raise ValueError('Duplicate Entries in Filter File')

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
AnnotFile = __ID_List_Factory__(None,2,None,header=0,usecols=None)



# def l1(fh, num=None):
#		'''
#		Parses .l1.ldscore files. See docs/file_formats_ld.txt
# 
#		If num is not None, parses .l1.ldscore files split across [num] chromosomes (e.g., the 
#		output of parallelizing ldsc.py --l1 across chromosomes).
# 
#		'''
#												
#		parsefunc = lambda y, compression : pd.read_csv(y, header=0, delim_whitespace=True,
#				compression=compression).drop(['CHR','BP','CM','MAF'], axis=1)
#		
#		if num is not None:
#				try:
#						suffix = '.l1.ldscore.gz'
#						if '@' in fh:
#								full_fh = fh.replace('@', '1') + suffix
#						else:
#								full_fh = fh + '1' + suffix
#		
#						x = open(full_fh, 'rb')
#						x.close()
#						compression = 'gzip'
#				except IOError:
#						suffix = '.l1.ldscore'
# 
#						if '@' in fh:
#								full_fh = fh.replace('@', '1') + suffix
#						else:
#								full_fh = fh + '1' + suffix
#						x = open(full_fh, 'rb')
#						x.close()
#						compression = None					
#				
#				if '@' in fh:
#						chr_ld = [parsefunc(fh.replace('@',str(i))+suffix, compression) for i in xrange(1,num+1)]
#				else:
#						chr_ld = [parsefunc(fh + str(i) + suffix, compression) for i in xrange(1,num+1)]
# 
#				x = pd.concat(chr_ld)
#		
#		else:
#				try:
#						full_fh = fh + '.l1.ldscore.gz'
#						open(full_fh, 'rb')
#						compression = 'gzip'
#				except IOError:
#						full_fh = fh + '.l1.ldscore'
#						open(full_fh, 'rb')
#						compression = None					
#				
#				x = parsefunc(full_fh, compression)
#		
#		ii = x['SNP'] != '.'
#		x = x[ii]		
#		check_rsid(x['SNP']) 
#		for col in x.columns[1:]:
#				x[col] = x[col].astype(float)
#		
#		return x

