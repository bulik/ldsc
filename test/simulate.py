'''
Generates .sumstats and .l2.ldscore/.l2.M files used for simulation testing.

'''

import numpy as np
import pandas as pd

N_INDIV = 10000
N_SIMS = 1000
N_SNP = 1000
h21 = 0.3
h22 = 0.6


def print_ld(x, fh, M):
    l2 = '.l2.ldscore'
    m = '.l2.M_5_50'
    x.to_csv(fh + l2, sep='\t', index=False, float_format='%.3f')
    print('\t'.join(map(str, M)), file=open(fh + m, 'wb'))

    # chr1
    y = x.iloc[0:int(len(x) / 2), ]
    y.to_csv(fh + '1' + l2, sep='\t', index=False, float_format='%.3f')
    print('\t'.join((str(x / 2) for x in M)), file=open(fh + '1' + m, 'wb'))

    # chr2
    y = x.iloc[int(len(x) / 2):len(x), ]
    y.to_csv(fh + '2' + l2, sep='\t', index=False, float_format='%.3f')
    print('\t'.join((str(x / 2) for x in M)), file=open(fh + '2' + m, 'wb'))

two_ldsc = np.abs(100 * np.random.normal(size=2 * N_SNP)).reshape((N_SNP, 2))
single_ldsc = np.sum(two_ldsc, axis=1).reshape((N_SNP, 1))
M_two = np.sum(two_ldsc, axis=0)
M = np.sum(single_ldsc)
ld = pd.DataFrame({
    'CHR': np.ones(N_SNP),
    'SNP': ['rs' + str(i) for i in range(1000)],
    'BP': np.arange(N_SNP)})

# 2 LD Scores 2 files
split_ldsc = ld.copy()
split_ldsc['LD'] = two_ldsc[:, 0]
print_ld(split_ldsc, 'simulate_test/ldscore/twold_firstfile', [M_two[0]])
split_ldsc = ld.copy()
split_ldsc['LD'] = two_ldsc[:, 1]  # both have same colname to test that this is ok
print_ld(split_ldsc, 'simulate_test/ldscore/twold_secondfile', [M_two[1]])

# 1 LD Score 1 file
ldsc = ld.copy()
ldsc['LD'] = single_ldsc
print_ld(ldsc, 'simulate_test/ldscore/oneld_onefile', [M])

# 2 LD Scores 1 file
ldsc = ld.copy()
ldsc['LD1'] = two_ldsc[:, 0]
ldsc['LD2'] = two_ldsc[:, 1]
print_ld(ldsc, 'simulate_test/ldscore/twold_onefile', M_two)

# Weight LD Scores
w_ld = ld.copy()
w_ld['LD'] = np.ones(N_SNP)
w_ld.to_csv('simulate_test/ldscore/w.l2.ldscore',
            index=False, sep='\t', float_format='%.3f')
# split across chromosomes
df = pd.DataFrame({
    'SNP': ['rs' + str(i) for i in range(1000)],
    'A1': ['A' for _ in range(1000)],
    'A2': ['G' for _ in range(1000)],
    'N': np.ones(1000) * N_INDIV
})
for i in range(N_SIMS):
    z = np.random.normal(size=N_SNP).reshape((N_SNP,))
    c = np.sqrt(
        1 + N_INDIV * (h21 * two_ldsc[:, 0] / float(M_two[0]) + h22 * two_ldsc[:, 1] / float(M_two[1])))
    z = np.multiply(z, c)
    dfi = df.copy()
    dfi['Z'] = z
    dfi.reindex(np.random.permutation(dfi.index))
    dfi.to_csv('simulate_test/sumstats/' + str(i),
               sep='\t', index=False, float_format='%.3f')
