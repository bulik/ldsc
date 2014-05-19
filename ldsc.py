from __future__ import division
import ldsc.ldscore as ld
import argparse
import numpy as np
import pandas as pd

def __filter__(fname, noun, verb, merge_obj):
	merged_list = None
	if fname:
		f = lambda x,n: x.format(noun=noun, verb=verb, fname=fname, num=n)
		x = ld.FilterFile(fname)
	 	c = 'Read list of {num} {noun} to {verb} from {fname}'
	 	print f(c, len(x.IDList))
		merged_list = merge_obj.loj(x.IDList)
		len_merged_list = len(merged_list)
		if len_merged_list > 0:
			c = 'After merging, {num} {noun} remain'
			print f(c, len_merged_list)
		else:
			error_msg = 'No {noun} retained for analysis'
			raise ValueError(f(error_msg))

		return merged_list


def ldscore(args):
	'''
	Annot format is 
	chr snp bp cm <annotations>
	'''

	if args.bin:
		snp_file, snp_obj = args.bin+'.snp', ld.VcfSNPFile
		ind_file, ind_obj = args.bin+'.ind', ld.VcfINDFile
		array_file, array_obj = args.bin+'.bin', ld.VcfBINFile
	elif args.bfile:
		snp_file, snp_obj = args.bfile+'.bim', ld.PlinkBIMFile
		ind_file, ind_obj = args.bfile+'.fam', ld.PlinkFAMFile
		array_file, array_obj = args.bfile+'.bed', ld.PlinkBEDFile

	# read bim/snp
	array_snps = snp_obj(snp_file)
	m = len(array_snps.IDList)
	print 'Read list of {m} SNPs from {f}'.format(m=m, f=snp_file)
	
	# read annot
	if args.annot:
		annot = ld.AnnotFile(args.annot)
		num_annots,ma = len(annot.df.columns) - 4, len(annot.df)
		print "Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,A=num_annots,
			M=ma)
		annot_matrix = np.array(annot.df.iloc[:,4:])
		annot_colnames = annot.df.columns[4:]
	else:
		array_keep, annot_matrix, annot_colnames = None, None, None
		num_annots = 1
	
	# read fam/ind
	array_indivs = ind_obj(ind_file)
	n = len(array_indivs.IDList)	 
	print 'Read list of {n} individuals from {f}'.format(n=n, f=ind_file)
	# read keep_indivs
	if args.extract:
		keep_indivs = __filter__(args.extract, 'individuals', 'include', array_indivs)
	else:
		keep_indivs = None
		
	# read genotype array
	print 'Reading genotypes from {fname}'.format(fname=array_file)
	geno_array = array_obj(array_file,n,array_snps, keep_indivs=keep_indivs, 
		mafMin=args.maf)
		
	# filter annot_matrix down to only SNPs passing MAF cutoffs
	if annot_matrix is not None:
		annot_keep = geno_array.kept_snps
		annot_matrix = annot_matrix[annot_keep,:]
	
	# determine block widths
	x = np.array((args.ld_wind_snps, args.ld_wind_kb, args.ld_wind_cm), dtype=bool)
	if np.sum(x) != 1: 
		raise ValueError('Must specify exactly one --ld-wind option')
	
	if args.ld_wind_snps:
		max_dist = args.ld_wind_snps
		coords = np.array(xrange(geno_array.m))
	elif args.ld_wind_kb:
		max_dist = args.ld_wind_kb*1000
		coords = np.array(array_snps.df['BP'])[geno_array.kept_snps]
	elif args.ld_wind_cm:
		max_dist = args.ld_wind_cm
		coords = np.array(array_snps.df['CM'])[geno_array.kept_snps]

	block_left = ld.getBlockLefts(coords, max_dist)
	if block_left[len(block_left)-1] == 0 and not args.yes_really:
		error_msg = 'Do you really want to compute whole-chomosome LD Score? If so, set the'
		error_msg += '--yes-really flag (warning: it will use a lot of time / memory)'
		raise ValueError(error_msg)

	if args.se: # block jackknife

		# block size
		if args.block_size:
			jSize = args.block_size 
		elif n > 50:
			jSize = 10
		else:
			jSize = 1
		
		jN = int(np.ceil(n / jSize))
		if args.l1:
			col_prefix = "L1"; file_suffix = "l1.jknife"
			raise NotImplementedError('Sorry, havent implemented L1 block jackknife yet.')
			
		elif args.l1sq:
			col_prefix = "L1SQ"; file_suffix = "l1sq.jknife"
			raise NotImplementedError('Sorry, havent implemented L1^2 block jackknife yet.')
			
		elif args.l2:
			col_prefix = "L2"; file_suffix = "l2.jknife"
			c = "Computing LD Score (L2) and block jackknife standard errors with {n} blocks."
			
		elif args.l4:
			col_prefix = "L4"; file_suffix = "l4.jknife"
			c = "Computing L4 and block jackknife standard errors with {n} blocks."
			
		print c.format(n=jN)
		(lN_est, lN_se) = geno_array.ldScoreBlockJackknife(block_left, args.chunk_size, jN=jN,
			annot=annot_matrix)
		lN = np.c_[lN_est, lN_se]
		if num_annots == 1:
			ldscore_colnames = [col_prefix, 'SE('+col_prefix+')']
		else:
			ldscore_colnames =  [x+col_prefix for x in annot_colnames]
			ldscore_colnames += ['SE('+x+')' for x in ldscore_colnames]

	else: # not block jackknife
		if args.l1:
			print "Estimating L1."
			lN = geno_array.l1VarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L1"; file_suffix = "l1"
		
		elif args.l1sq:
			print "Estimating L1 ^ 2."
			lN = geno_array.l1sqVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L1SQ"; file_suffix = "l1sq"
		
		elif args.l2:
			print "Estimating LD Score (L2)."
			lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
			col_prefix = "L2"; file_suffix = "l2"
	
		elif args.l4:
			col_prefix = "L4"; file_suffix = "l4"
			raise NotImplementedError('Sorry, havent implemented L4 yet. Try the jackknife.')
			lN = geno_array.l4VarBlocks(block_left, c, annot)
		
		if num_annots == 1:
			ldscore_colnames = [col_prefix]
		else:
			ldscore_colnames =  [x+col_prefix for x in annot_colnames]
			
	# print .ldscore
	# output columns: CHR, BP, CM, RS, MAF, [LD Scores and optionally SEs]
	out_fname = args.out + '.' + file_suffix + '.ldscore'
	new_colnames = geno_array.colnames + ldscore_colnames
	df = pd.DataFrame(np.c_[geno_array.df, lN])
	df.columns = new_colnames
	print "Writing results to {f}".format(f=out_fname)
	df.to_csv(out_fname, sep="\t", header=True, index=False)	
	
	# print .M
	fout_M = open(args.out + '.'+ file_suffix +'.M','wb')
	if num_annots == 1:
		print >> fout_M, geno_array.m
	else:
		M = np.sum(annot_matrix, axis=0)
		print >> fout_M, '\t'.join(map(str,M))

	fout_M.close()
	
def simulate():
	pass

def estimate():
	pass
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
		
	# LD Score Estimation Arguments
	
	# Input
	parser.add_argument('--bin', default=None, type=str, 
		help='Prefix for binary VCF file')
	parser.add_argument('--bfile', default=None, type=str, 
		help='Prefix for Plink .bed/.bim/.fam file')
	parser.add_argument('--annot', default=None, type=str, 
		help='Filename prefix for annotation file for partitioned LD Score estimation')

	# Data management
	parser.add_argument('--extract', default=None, type=str, 
		help='File with individuals to include in LD Score analysis')
	parser.add_argument('--maf', default=0, type=float,
		help='Minor allele frequency lower bound. Default is 0')
	parser.add_argument('--ld-wind-snps', default=None, type=int,
		help='LD Window in units of SNPs. Can only specify one --ld-wind-* option')
	parser.add_argument('--ld-wind-kb', default=None, type=float,
		help='LD Window in units of kb. Can only specify one --ld-wind-* option')
	parser.add_argument('--ld-wind-cm', default=None, type=float,
		help='LD Window in units of cM. Can only specify one --ld-wind-* option')
	parser.add_argument('--chunk-size', default=50, type=int,
		help='Chunk size for LD Score calculation. Use the default')

	# Output
	parser.add_argument('--l1', default=False, action='store_true',
		help='Estimate l1 w.r.t. sample minor allele.')
	parser.add_argument('--l1sq', default=False, action='store_true',
		help='Estimate l1 ^ 2 w.r.t. sample minor allele.')
	parser.add_argument('--l2', default=False, action='store_true',
		help='Estimate l2. Compatible with both jackknife and non-jackknife.')
	parser.add_argument('--l4', default=False, action='store_true',
		help='Estimate l4. Only compatible with jackknife.')
	
	parser.add_argument('--se', action='store_true', 
		help='Block jackknife SE? (Warning: somewhat slower)')
	parser.add_argument('--block-size', default=None, type=int, 
		help='Block size for block jackknife')
	parser.add_argument('--yes-really', default=False, action='store_true',
		help='Yes, I really want to compute whole-chromosome LD Score')
	parser.add_argument('--out', default='ldsc', type=str,
		help='Output filename prefix')
	
	# Simulation Arguments
	# Summary Statistic Estimation Arguments
	
	args = parser.parse_args()
	
	if args.bin or args.bfile:
		ldscore(args)
		