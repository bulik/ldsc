'''
Generates .sumstats and .l2.ldscore/.l2.M files used for simulation testing.

'''
from __future__ import division
import numpy as np
import pandas as pd

N_SIMS = 1000
N_SNP = 1000

two_ldsc = np.abs(100*np.random.normal(size=2*N_SNP)).reshape((N_SNP, 2))
single_ldsc = np.sum(two_ldsc, axis=1).reshape((N_SNP, 1))
ld = pd.DataFrame({
	'CHR': np.ones(N_SNP),
	'SNP': ['rs'+str(i) for i in xrange(1000)],
	'BP': np.arange(N_SNP),
	'CM': np.zeros(N_SNP),
	'MAF': np.ones(N_SNP)/2})

def print_ld(x, fh, M):
	l2 = '.l2.ldscore'
	m = '.l2.M_5_50'
	x.to_csv(fh+l2, sep='\t', index=False)
	print >>open(fh+m, 'wb'), '\t'.join(map(str, M))

	# chr1
	y = x.iloc[0:int(len(x)/2),]
	y.to_csv(fh+'1'+l2, sep='\t', index=False)
	print >>open(fh+'1'+m, 'wb'), '\t'.join((str(x/2) for x in M))
	
	# chr2
	y = x.iloc[int(len(x)/2):len(x),]
	y.to_csv(fh+'2'+l2, sep='\t', index=False)
	print >>open(fh+'2'+m, 'wb'), '\t'.join((str(x/2) for x in M))
		
	
# 2 LD Scores 2 files
split_ldsc = ld.copy()
split_ldsc['LD1'] = two_ldsc[:,0]
print_ld(split_ldsc, 'simulate_test/ldscore/twold_firstfile', [4000])
split_ldsc = ld.copy()
split_ldsc['LD2'] = two_ldsc[:,1]
print_ld(split_ldsc, 'simulate_test/ldscore/twold_secondfile', [8000])

# 1 LD Score 1 file
ldsc = ld.copy()
ldsc['LD'] = single_ldsc
print_ld(ldsc, 'simulate_test/ldscore/oneld_onefile', [12000])

# 2 LD Scores 1 file
ldsc = ld.copy()
ldsc['LD1'] = two_ldsc[:,0]
ldsc['LD2'] = two_ldsc[:,1]
print_ld(ldsc, 'simulate_test/ldscore/twold_onefile', [4000, 8000])


# Weight LD Scores
w_ld = ld.copy()
w_ld['LD'] = np.ones(N_SNP)
w_ld.to_csv('simulate_test/ldscore/w.l2.ldscore', index=False, sep='\t')
# split across chromosomes
df = pd.DataFrame({
	'SNP': ['rs'+str(i) for i in xrange(1000)],
	'A1' : ['A' for _ in xrange(1000)],
	'A2' : ['G' for _ in xrange(1000)],
	'N': np.ones(1000)*1000
})
for i in xrange(N_SIMS):
	z = np.random.normal(size=N_SNP).reshape((N_SNP, 1))
	z = np.multiply(z, np.sqrt(single_ldsc))
	dfi = df.copy()
	dfi['BETA'] = z/1000
	dfi.reindex(np.random.permutation(dfi.index))
	dfi.to_csv('simulate_test/sumstats/'+str(i), sep='\t', index=False)